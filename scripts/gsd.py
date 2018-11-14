#!/usr/bin/env python3

# OS
import os
import pickle
import multiprocessing
import logging

# Data processing
import pandas as pd
import numpy as np

# Networks
import networkx as nx
from pcst_fast import pcst_fast

# Statistics
from scipy import stats
from sklearn.decomposition import FastICA


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(asctime)s - GSD: %(levelname)s - %(message)s', "%I:%M:%S"))
logger.addHandler(handler)


n_cpus = multiprocessing.cpu_count()

class GSD:

	def __init__(self, X, graph, a, n_components, initializer='random'): 

		# Import as networkx object if path is provided
		if isinstance(graph, str): graph = nx.read_gpickle(graph)

		self.X = X          # Data matrix (n_samples, n_features) whose features are ordered according to graph.nodes
		self.graph = graph  # networkx interactome
		self.a = a          # Sparsity edge constraint 
		self.n_components = n_components
		self.n_samples, self.n_features = self.X.shape

		if self.n_samples > self.n_features: 
			logger.warn("Number of samples exceeds number of features, class may not function properly!")

		# Get graph edgelist as node indices for PCST.
		self._set_namespace_mappings()
		self._edges_pcst = self._to_indices(self.graph.edges)
		# Get scaled edge costs.
		self.edge_cost = self._edge_costs()

		# Initialize graph components.
		if initializer == 'ica': 
			self.components, self.scores = self._ica_initializer()
		elif initializer == 'random': 
			self.components, self.scores = self._random_initializer()
		else: 
			logger.warn("Initializer must be set to `random` or `ica`.")

		# Initialize edges. Finds the minimum spanning tree to use as a baseline.
		self.tree_edges = [ self._pcst(np.ones(self.n_features), self.edge_cost*0)[1] ] * self.n_components
		# Initialize objective scores
		self.objective, self.error, self.tree_cost = self.compute_objective(self.components, self.scores, self.tree_edges)


	########  HELPER FUNCTIONS  ########

	def _set_namespace_mappings(self): 
		"""Sets namespace mappings between gene names and node indices."""
		self._gene_to_index = { gene: i for i,gene in enumerate(self.graph) }
		self._index_to_gene = { i: gene for i,gene in enumerate(self.graph) }
	
	def _to_indices(self, genes): 
		"""Converts object with gene names to node indices."""
		return np.vectorize(self._gene_to_index.get)(genes)
		
	def _to_genes(self, indices): 
		"""Converts object with node indices to gene names."""
		return np.vectorize(self._index_to_gene.get)(indices)

	def _pcst(self, prizes, costs): 
		"""Wrapper for `pcsf_fast`. Returns vertex and edge indices."""
		return pcst_fast(self._edges_pcst, prizes, costs, -1, 1, 'strong', 0)
		
	def _edge_costs(self): 
		"""Returns edge costs scaled by `a`."""
		return self.a * np.array([ cost for _,_,cost in self.graph.edges.data('cost') ])

	def supports(self): 
		"""Gets component supports."""
		return (self.components != 0).astype(int)


	########  DICTIONARY LEARNING  ########

	def update(self, i, same_sign=True): 
		"""
		Updates component and scores matrix together along with associated tree_edgelist and model metrics

		Arguments: 
			i (int): component index
			same_sign (bool): if True, component values must be all positive or all negative
		"""
		new_D, new_tree_i = self._update_component(i, self.components, self.scores, same_sign=same_sign)
		new_Z = self._update_scores(new_D)

		# Check if updated matrices improve upon objective.
		if True: 
			self.components = new_D
			self.scores = new_Z
			self.tree_edges[i] = new_tree_i

			self.objective, self.error, self.tree_cost = self.compute_objective(self.components, self.scores, self.tree_edges)

		# Check if new component is too small or only one element, then remove. 
		# TODO

	def _update_component(self, i, D, Z, same_sign=True): 
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

		# Calculate optimal D_i via least squares estimation (in other words, what's the 
		# best component assuming no sparsity constraint)
		Z_i = Z[:,i]
		optimal_D_i = np.dot(Z_i, Z_i) ** (-1) * np.dot(Z_i, X_res)

		# Assign prizes to be the amount of error saved by including a node in the component tree. 
		# This is calculated by taking the difference in mean reconstruction error with and without the node.
		prizes = (X_res ** 2).mean(axis=0) - ((X_res - np.outer(Z_i, optimal_D_i)) ** 2).mean(axis=0)

		if same_sign: 
			# Require values within each component to have the same sign by setting values of the opposite sign to 0.
			if stats.skew(optimal_D_i) > 0:
				prizes[ optimal_D_i < 0 ] = 0
			else: 
				prizes[ optimal_D_i > 0 ] = 0

		node_indices, edge_indices = self._pcst(prizes, self.edge_cost)

		# Replace component i with sparsified optimal D_i component specified by PCST results
		D[i] = np.array([ optimal_D_i[i] if i in node_indices else 0 for i in range(self.n_features) ])

		return D, edge_indices

	def _update_scores(self, D): 
		"""Updates scores matrix by fixing dictionary D."""
		return self._OLS(self.X, D)

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
	
	def compute_objective(self, D, Z, tree_edges): 
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
		tree_cost = sum([ sum(self.edge_cost[edges]) for edges in tree_edges ])
		# Reconstruction error
		error = ((self.X - Z.dot(D)) ** 2).sum()
		# Objective score
		objective = error + tree_cost
		
		return objective, error, tree_cost


	########  INITIALIZATION  ########
	
	def _random_initializer(self): 
		"""Initializes dictionary and scores matrices randomly."""
		components = np.random.normal( size=(self.n_components, self.n_features) ) 
		scores     = self._OLS( self.X, components ) # (n_samples, n_components)
		logger.info("Initialized components and scores randomly.")
		
		return components, scores


	def _ica_initializer(self): 
		"""Initializes dictionary and scores matrices with independent component analysis (ICA)."""
		ica = FastICA(n_components=self.n_components, max_iter=1000)
		# Because n_samples < n_features, we must transpose
		sources = ica.fit_transform(self.X.T).T  # (n_sources, n_features)
		mixing  = ica.mixing_  # (n_samples, n_sources)
		logger.info("Initialized components and scores via ICA.")

		return sources, mixing

