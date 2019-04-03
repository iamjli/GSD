#!/usr/bin/env python3

# Core python modules
import os
import pickle
import multiprocessing
import logging
import options

# Python external libraries
import pandas as pd
import numpy as np
import networkx as nx
from scipy import stats
from sklearn.decomposition import PCA, FastICA

# Lab modules
from pcst_fast import pcst_fast


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(asctime)s - GSD: %(levelname)s - %(message)s', "%I:%M:%S"))
logger.addHandler(handler)


# n_cpus = multiprocessing.cpu_count()


class GSD:

	defaults = options.Options(
		a = 0.4, 
		w = 1, 
		n_components = 3, 
		n_iterations = 5,

		method      = "gsPCA", 
		initializer = "zeros", 
		same_sign   = True, 

		dummy_nodes = [], 
		rooted_PCSF = False
	)

	def __init__(self, data, edgelist, params=dict()): 

		self.params = GSD.defaults.push(params)

		self.data = data
		self.edgelist = edgelist

		self._initialize_indices()
		self._initialize_matrices()

		# Initialize edges. Finds the minimum spanning tree to use as a baseline.
		self.D_edges = [ self._pcsf( prizes=np.ones(self.n_features), costs=self.costs*0 )[1] ] * self.n_components
		# Initialize objective scores
		self.objective, self.error, self.D_cost = self.compute_objective(self.D, self.Z, self.D_edges)


	########  INITIALIZATION  ########

	def _initialize_indices(self): 

		# Convert the interactome dataframe from string interactor IDs to integer interactor IDs.
		# Do so by selecting the protein1 and protein2 columns from the interactome dataframe,
		# then unstacking them, which (unintuitively) stacks them into one column, allowing us to use factorize.
		# Factorize builds two datastructures, a unique pd.Index which maps each ID string to an integer ID,
		# and the datastructure we passed in with string IDs replaced with those integer IDs.
		# We place those in self.nodes and self.edges respectively, but self.edges will need reshaping.
		(self.edges, self.nodes) = pd.factorize(self.edgelist[["anchor1","anchor2"]].unstack())
		# Here we do the inverse operation of "unstack" above, which gives us an interpretable edges datastructure
		self.edges = self.edges.reshape(self.edgelist[["anchor1","anchor2"]].shape, order='F')

		self.costs = self.edgelist["cost"].astype(float).values

		# Add dummy nodes. If `self.params.dummy_nodes` is empty list, these edges will be unchanged. 
		endpoints = np.where(self.nodes.isin(self.params.dummy_nodes))[0]
		self._dummy_id = len(self.nodes)
		if len(endpoints) > 0: 
			# Add edges between dummy node and each endpoint
			self.edges = np.concatenate((self.edges, np.array([(self._dummy_id, node_id) for node_id in endpoints])))
			# Assign costs of dummy edges to be w
			self.costs = np.concatenate((self.costs, np.array([self.params.w] * len(endpoints))))

	def _initialize_matrices(self): 

		# Reindex new dataframe to match node indices
		self.data = self.data.reindex(columns=self.nodes).fillna(0)
		self.X = self.data.values

		(self.n_samples, self.n_features) = self.data.shape
		self.n_components = self.params.n_components

		# Initialize submatrices
		if self.params.initializer == "random": 
			self.D = np.random.normal(size=(self.n_components, self.n_features)) 
			self.Z = self._update_Z(self.D)

		elif self.params.initializer == "zeros": 
			self.D = np.zeros(shape=(self.n_components, self.n_features)) 
			self.Z = self._update_Z(self.D) 

		elif self.params.initializer == "ica": 
			ica = FastICA(n_components=self.n_components)
			self.D = ica.fit_transform(self.X.T).T  
			self.Z = ica.mixing_ 

		else: 
			pass


	########  DICTIONARY LEARNING  ########

	def _pcsf(self, prizes, costs):

		if len(self.params.dummy_nodes) == 0: 
			# Unrooted PCST
			node_indices, edge_indices = pcst_fast(self.edges, prizes, costs, -1, 1, "strong", 0)

		else: 
			if self.params.rooted_PCSF: 
				# Classic implementation of PCSF which roots the tree at the dummy node which has a prize of 0. 
				# See OI2 for more details. 
				prizes = np.concatenate((prizes, [0]))
				node_indices, edge_indices = pcst_fast(self.edges, prizes, costs, self._dummy_id, 1, "strong", 0)

			else: 
				# Alternative implementation of PCSF that artificially roots the dummy node by assigning
				# a prize to it such that the prize is slightly larger than the cost of its associated edges. 
				# This method of rooting runs faster than rooting the graph explicitly using the 
				# `root` parameter in the `pcst_fast` function, but may offer worse solutions. 
				prizes = np.concatenate((prizes, [ max(costs) * 1.01 ]))
				node_indices, edge_indices = pcst_fast(self.edges, prizes, costs, -1, 1, "strong", 0)

			# Remove references to dummy node
			node_indices = node_indices[node_indices != self._dummy_id]
			edge_indices = self.edgelist.index & edge_indices

		return node_indices, edge_indices

	def update(self, i): 
		"""
		Updates component and scores matrix together along with associated tree_edgelist and model metrics

		Arguments: 
			i (int): component index
			same_sign (bool): if True, component values must be all positive or all negative
		"""
		new_D, new_D_edges = self._update_D(i, self.D, self.Z)
		new_Z = self._update_Z(new_D)

		# Check if updated matrices improve upon objective.
		if True: 
			self.D = new_D
			self.Z = new_Z
			self.D_edges[i] = new_D_edges

			self.objective, self.error, self.D_cost = self.compute_objective(self.D, self.Z, self.D_edges)

	def _update_D(self, i, D, Z): 
		"""
		Updates component i of dictionary D while fixing the rest of the components and scores Z to be constant.

		First, contributes from all other components besides i are subtracted to obtain a residual for i 
		on each node. Note that optimal values for each node in the component can be obtained individually
		via least squares estimation. The difference between the residual remaining and the residual in the 
		optimal case is used as prizes for PCST. 

		Arguments: 
			i (int): component index
			D (numpy.array): 2D dictionary components array (n_components, n_features) 
			Z (numpy.array): 2D scores array (n_samples, n_components)
			same_sign (bool): if True, component values must be all positive or all negative

		Returns: 
			numpy.array: updated 2D dictionary components array (n_components, n_features) 
			numpy.array: indices of tree associated with component i
		"""

		# Get residual with component i held out
		X_res = self.X - np.dot(np.delete(Z, i, axis=1), np.delete(D, i, axis=0))

		if self.params.method == 'dict_learning': 
			# Calculate optimal D_i via least squares estimation (in other words, what's the 
			# best component assuming no sparsity constraint)
			Z_i = Z[:,i]
			optimal_D_i = np.dot(Z_i, Z_i) ** (-1) * np.dot(Z_i, X_res)
		elif self.params.method == 'gsPCA': 
			# Model component after graph-sparse PC1
			pca = PCA(n_components=1)
			Z_i = pca.fit_transform(X_res).T[0]
			optimal_D_i = pca.components_[0]

		# Ensures that component has positive skew (easier for component interpretation)
		if stats.skew(optimal_D_i) < 0: 
			optimal_D_i *= -1
			Z_i *= -1

		# Assign prizes to be the amount of error saved by including a node in the component tree. 
		# This is calculated by taking the difference in mean reconstruction error with and without the node.
		prizes = (X_res ** 2).mean(axis=0) - ((X_res - np.outer(Z_i, optimal_D_i)) ** 2).mean(axis=0)

		if self.params.same_sign: 
			# Require values within each component to have the same sign by setting values of the opposite sign to 0.
			prizes[ optimal_D_i < 0 ] = 0

		# In theory, prizes should be non-negative but we clip in case of floating point errors
		# prizes[ abs(prizes) < 1e-8 ] = 0
		prizes = prizes.clip(min=0) 

		node_indices, edge_indices = self._pcsf(prizes, self.costs * self.params.a)

		if len(node_indices) <= 1: 
			logger.warn("Component {} is deprecated!".format(i))

		# Replace component i with sparsified optimal D_i component specified by PCST results
		D[i] = np.array([ optimal_D_i[i] if i in node_indices else 0 for i in range(self.n_features) ])
		D[i][ abs(D[i]) < 1e-8 ] = 0

		return D, edge_indices


	def _update_Z(self, D): 
		"""Updates scores matrix by fixing dictionary D."""
		Z = np.zeros( shape=(self.n_samples, self.n_components) )
		# Component row indices that are not empty
		nonempty_indices = np.where(D.any(axis=1))[0]
		Z[:,nonempty_indices] = self._OLS(self.X, D[nonempty_indices])
		return Z

	def _OLS(self, X, D):
		"""
		Ordinary least squares - computes scores matrix Z that minimizes X ~ Z . D

		Arguments: 
			X (numpy.array): 2D data matrix (n_samples, n_features) 
			D (numpy.array): 2D dictionary components array (n_components, n_features) 

		Returns: 
			numpy.array: 2D scores array indicating proportions of each component (n_samples, n_components)
		"""
		return np.linalg.inv(D.dot(D.T)).dot(D).dot(X.T).T


	########  CONVENIENCE  ########

	def compute_objective(self, D, Z, D_edges): 
		"""
		Computes objective score for components D and scores Z

		Arguments: 
			D (numpy.array): 2D dictionary components array (n_components, n_features) 
			Z (numpy.array): 2D scores array (n_samples, n_components)
			tree_edges (list of lists): Edge indices corresponding to each component in D

		Returns: 
			float: objective score
			float: reconstruction error 
			float: cost of all trees
		"""
		
		# Total cost of edges for all trees
		D_cost = sum([ sum(self.costs[edges]) for edges in D_edges ]) * self.params.a
		# Reconstruction error
		error = ((self.X - Z.dot(D)) ** 2).sum() / self.n_samples
		# Objective score
		objective = error + D_cost
		
		return objective, error, D_cost


	def get_steiner_support_df(self): 
		"""Gets components that may contain Steiner nodes."""
		supports = np.zeros(shape=(self.n_components, self.n_features), dtype=int)
		for i,edges in enumerate(self.D_edges): 
			supports[i,np.unique(self.edges[edges])] = 1
		supports = pd.DataFrame(supports, columns=self.data.columns)
		return supports


	def get_edge_support_df(self): 
		membership = np.zeros(shape=(self.n_components, len(self.edgelist)), dtype=int)
		for i,edges in enumerate(self.D_edges): 
			membership[i,edges] = 1
		membership = pd.DataFrame(membership)
		return membership


	def output_normalized_decomposition(self): 

		D_scale = (self.D ** 2).sum(axis=1) ** 0.5
		D_norm = self.D * np.expand_dims(1/D_scale, axis=1)
		component_norm = pd.DataFrame(D_norm, columns=self.data.columns)

		Z_scale = (self.Z ** 2).sum(axis=0) ** 0.5
		Z_norm = self.Z * np.expand_dims(1/Z_scale, axis=0)
		score_norm = pd.DataFrame(Z_norm, index=self.data.index)

		singular_values = D_scale * Z_scale

		# X ~ (normalized_scores * singular_values).dot(normalized_components)

		return score_norm, singular_values, component_norm

