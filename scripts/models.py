
# OS
import os
import pickle
import multiprocessing

# Data processing
import pandas as pd
import numpy as np

# Networks
import networkx as nx
from pcst_fast import pcst_fast

# Statistics
from scipy import stats
from sklearn.decomposition import PCA, FastICA



n_cpus = multiprocessing.cpu_count()


class GSD: 

	def __init__(self, X, graph, a, K, initializer='random'): 

		self.X = X
		self.graph = graph
		self.a = a
		self.K = K 

		# Mappings. Enumerating through network object iterates through its nodes
		self._gene_to_index = { gene: i for i,gene in enumerate(self.graph) }
		self._index_to_gene = { i: gene for i,gene in enumerate(self.graph) }

		# Get PCST edgelist and gene list
		self._edges_pcst = self._to_indices(self.graph.edges)
		
		# Get edge costs as array
		self.edge_cost = self.get_edge_costs(self.a)

		# Initialize graph components
		if initializer == 'svd': 
			self.scores, self.components = self._svd_initializer(self.X, n_components=self.K)
		elif initializer == 'random': 
			self.components, self.scores = self._random_initializer(self.X, n_components=self.K)

		# Initialize edges. Finds the minimum spanning tree to use as a baseline.
		self.tree_edges = [self._initialize_trees()] * self.K
		
		self.objective, self.error, self.tree_cost = self.compute_objective(self.components, self.scores, self.tree_edges)


	########  HELPER FUNCTIONS  ########
	
	def _pcst(self, prizes, costs): 
		# Wrapper for `pcsf_fast`. Returns vertex and edge indices
		return pcst_fast(self._edges_pcst, prizes, costs, -1, 1, 'strong', 0)
	
	def _to_indices(self, genes): 
		return np.vectorize(self._gene_to_index.get)(genes)
		
	def _to_genes(self, indices): 
		return np.vectorize(self._index_to_gene.get)(indices)
		
	def get_edge_costs(self, a): 

		cost = list(nx.get_edge_attributes(self.graph, 'cost').values())
		return np.array(a * cost)

	def _initialize_trees(self): 
		
		_,edge_indices = self._pcst(np.ones(len(self.graph.nodes)), np.zeros(len(self.graph.edges)))
		return edge_indices


	########  DICTIONARY LEARNING  ########
	
	def compute_objective(self, D, Z, tree_edges): 
		
		# Residual of input
		X_res = self.X - D.dot(Z)
		# Total cost of edges for all trees
		tree_cost = sum([sum(self.edge_cost[edges]) for edges in tree_edges])
		# Reconstruction error
		error	 = (X_res ** 2).sum()
		objective = error + tree_cost
		
		return objective, error, tree_cost
	
	def OLS(self, D):
		# Hold dictionary constant, optimize scores matrix
		# Dictionary is genes by K
		# Returns scores which should be K by n_samples
		return np.linalg.inv(D.T.dot(D)).dot(D.T).dot(self.X)
	
	def update_component(self, i): 
		
		# Fix scores matrix and all but one component, D_i, constant
		# Compute outer products of each score vector and its corresponding component. The sum of 
		# these outer products are fixed, and we subtract it from X.
		outer_products = [np.outer(component,score) for component,score in zip(self.components.T, self.scores)]
		X_res = self.X - sum([prod for i_vec,prod in enumerate(outer_products) if i_vec != i])
		
		# Calculate optimal dictionary values using least squared estimate again
		Z_i = self.scores[i]
		optimal_D_i = (Z_i.dot(Z_i)) ** (-1) * Z_i.dot(X_res.T)

		
		##### TODO!! 
		##### X was previously samples by genes, but needs to be changed to genes by samples
		##### The remaining function needs to be updated

		# Prizes are amount of error saved by including node in tree. Calculate by computing 
		# the matrix if no nodes are included and the matrix if optimal nodes are included, the taking the difference.
		prizes = (X_res ** 2).sum(axis=0) - ((X_res - np.outer(Z_i, optimal_D_i)) ** 2).sum(axis=0)
		# Here we make the assumption that the loadings of each module should overall be 
		# consistent in sign (all up or all down). We first find the skew in the D_i vector, then 
		# set all in the opposite sign to be 0. 
		if stats.skew(optimal_D_i) > 0: prizes = (optimal_D_i > 0).astype(int) * prizes
		else: prizes = (optimal_D_i < 0).astype(int) * prizes
		# Prize should be stricly positive, but we clip to correct for possible negatives caused by floating points
		prizes = prizes.clip(min=0)
		
		# Run PCST
		node_indices, edge_indices = self._pcst(prizes, self.edge_cost)
		# Asign optimal D_i values to sparsified signal
		sparse_signal = np.array([optimal_D_i[i] if i in node_indices else 0 for i in range(len(prizes))])
		
		# Assign new component if objective is reduced
		# In practice, objective never increases, so we can skip this check.
#		 self.components[i] = sparse_signal
#		 self.tree_edges[i] = edge_indices
#		 self.scores		= self.OLS(self.components)
#		 self.objective	 = self.compute_objective(self.components, self.scores, self.tree_edges)

		# In practice, objective never increases, so we can skip this check.
		components = self.components.copy()
		components[i] = sparse_signal
		scores = self.OLS(components)
		tree_edges = self.tree_edges.copy()
		tree_edges[i] = edge_indices
		
		objective, error, tree_cost = self.compute_objective(components, scores, tree_edges)
		if objective < self.objective:
			self.components = components
			self.scores	 = scores
			self.tree_edges = tree_edges
			self.objective  = objective
			self.error	  = error
			self.tree_cost  = tree_cost


	########  INITIALIZATION  ########
	
	def _random_initializer(self, X_input, n_components): 
		
		# Components - genes by K
		components = np.random.normal(size=(len(X_input), n_components))
		# Scores - K by samples
		scores = self.OLS(components)
		
		return components, scores














class Decomposition: 
	
	def __init__(self, X, G, g, a, K, initializer='svd'): 
		
		self.X_samples = X.index
		self.X_genes   = X.columns
		
		self.X = X.values  # input matrix (L,N): Rows are L samples, columns are N genes
		self.G = G		 # interactome
		self.g = g		 # hub node penalization term
		self.a = a		 # sparsity term
		self.K = K		 # number of components
		
		self.nodes = list(self.G.nodes)
		self.edges = list(self.G.edges)
		
		# Mappings. For `index_to_gene`, `self.nodes` is already indexed correctly. 
		self._gene_to_index = { gene: i for i,gene in enumerate(self.nodes) }
		self._index_to_gene = { i: gene for i,gene in enumerate(self.nodes) }
		
		# Get PCST edgelist and gene list
		self._edges_pcst = self._to_indices(self.edges)
		
		# Get edge costs as array
		self.edge_penalties = self._get_edge_penalty()
		self.edge_cost = self.get_edge_costs(self.g, self.a)
		
		# Initialize graph components
		if initializer == 'svd': 
			self.scores, self.components = self._svd_initializer(self.X, n_components=self.K)
		elif initializer == 'random': 
			self.scores, self.components = self._random_initializer(self.X, n_components=self.K)
		
		# Initialize edges. Finds the minimum spanning tree to use as a baseline.
		self.tree_edges = [self._initialize_trees()] * self.K
		
		self.objective, self.error, self.tree_cost = self.compute_objective(self.components, self.scores, self.tree_edges)
		
		
	########  HELPER FUNCTIONS  ########
	
	def _pcst(self, prizes, costs): 
		# Wrapper for `pcsf_fast`. Returns vertex and edge indices
		return pcst_fast(self._edges_pcst, prizes, costs, -1, 1, 'strong', 0)
	
	def _to_indices(self, genes): 
		return np.vectorize(self._gene_to_index.get)(genes)
		
	def _to_genes(self, indices): 
		return np.vectorize(self._index_to_gene.get)(indices)

	def _get_edge_penalty(self): 
		
		N = len(self.nodes)
		degrees = dict(self.G.degree(weight='cost'))
		edge_penalties = np.array([degrees[a]*degrees[b] / ((N-degrees[a]-1) * (N-degrees[b]-1) + degrees[a]*degrees[b]) for a,b in self.G.edges])
		
		return edge_penalties
		
	def get_edge_costs(self, g, a): 
		# Add edge penalty to edge costs
		# TODO: add method to not penalize edges at all
		cost = list(nx.get_edge_attributes(self.G, 'cost').values())
		if g == False:
			return np.array(a * cost)
		else:
			return a * (cost + (10**g) * self.edge_penalties)

	
	def _initialize_trees(self): 
		
		_,edge_indices = self._pcst(np.ones(len(self.nodes)), np.zeros(len(self.edges)))
		
		return edge_indices
	
	
	########  DICTIONARY LEARNING  ########
	
	def compute_objective(self, D, Z, tree_edges): 
		
		# Residual of input
		X_res	 = self.X - Z.dot(D)
		# Total cost of edges for all trees
		tree_cost = sum([sum(self.edge_cost[edges]) for edges in tree_edges])
		# Reconstruction error
		error	 = (X_res ** 2).sum()
		objective = error + tree_cost
		
		return objective, error, tree_cost
	
	def OLS(self, D):
		# Hold dictionary constant, optimize scores matrix
		return np.linalg.inv(D.dot(D.T)).dot(D).dot(self.X.T).T
	
	def update_component(self, i): 
		
		# Fix scores matrix and all but one component, D_i, constant
		# Compute outer products of each score vector and its corresponding component. The sum of 
		# these outer products are fixed, and we subtract it from X.
		outer_products = [np.outer(score,component) for score,component in zip(self.scores.T, self.components)]
		X_res = self.X - sum([prod for i_vec,prod in enumerate(outer_products) if i_vec != i])
		
		# Calculate optimal dictionary values using least squared estimate again
		Z_i = self.scores[:, i]
		optimal_D_i = (Z_i.dot(Z_i)) ** (-1) * Z_i.dot(X_res)
		
		# Prizes are amount of error saved by including node in tree. Calculate by computing 
		# the matrix if no nodes are included and the matrix if optimal nodes are included, the taking the difference.
		prizes = (X_res ** 2).sum(axis=0) - ((X_res - np.outer(Z_i, optimal_D_i)) ** 2).sum(axis=0)
		# Here we make the assumption that the loadings of each module should overall be 
		# consistent in sign (all up or all down). We first find the skew in the D_i vector, then 
		# set all in the opposite sign to be 0. 
		if stats.skew(optimal_D_i) > 0: prizes = (optimal_D_i > 0).astype(int) * prizes
		else: prizes = (optimal_D_i < 0).astype(int) * prizes
		# Prize should be stricly positive, but we clip to correct for possible negatives caused by floating points
		prizes = prizes.clip(min=0)
		
		# Run PCST
		node_indices, edge_indices = self._pcst(prizes, self.edge_cost)
		# Asign optimal D_i values to sparsified signal
		sparse_signal = np.array([optimal_D_i[i] if i in node_indices else 0 for i in range(len(prizes))])
		
		# Assign new component if objective is reduced
		# In practice, objective never increases, so we can skip this check.
#		 self.components[i] = sparse_signal
#		 self.tree_edges[i] = edge_indices
#		 self.scores		= self.OLS(self.components)
#		 self.objective	 = self.compute_objective(self.components, self.scores, self.tree_edges)

		# In practice, objective never increases, so we can skip this check.
		components = self.components.copy()
		components[i] = sparse_signal
		scores = self.OLS(components)
		tree_edges = self.tree_edges.copy()
		tree_edges[i] = edge_indices
		
		objective, error, tree_cost = self.compute_objective(components, scores, tree_edges)
		if objective < self.objective:
			self.components = components
			self.scores	 = scores
			self.tree_edges = tree_edges
			self.objective  = objective
			self.error	  = error
			self.tree_cost  = tree_cost

	
	########  INITIALIZATION  ########
	
	def _random_initializer(self, X_input, n_components='full'): 
		
		components = np.random.normal(size=X_input.shape)
		scores = self.OLS(components)
		
		if n_components == 'full': return scores, components
		
		return scores[:, :n_components], components[:n_components]  # (L, n_components), (n_components, N)
	
	
	def _svd_initializer(self, X_input, n_components='full'): 
		
		# X_input: input data matrix (L,N) with columns being N genes (features) and rows being L samples, N > L
		
		# U: Left singular matrix (L,L) with columns indicating relative weights of each component
		# S: Singular values (L,) indicating importance of each component
		# components: Right singular matrix (L,N) with each row corresponding to a component
		U, S, components = np.linalg.svd(X_input, full_matrices=False) 
		self.singular_values = S
		# scores: Score matrix (L x L) with columns indicating weights of each component such that 
		# X_input = scores . components
		scores = U * S
		
		if n_components == 'full': return scores, components
		
		return scores[:, :n_components], components[:n_components]  # (L, n_components), (n_components, N)
	

	########  ANALYTICS  ########
	def print_costs(self): 
		print("{0:.2f}\t{0:.2f}\t{0:.2f}".format(self.objective, self.error, self.tree_cost))


	def get_component_as_networkx(self, i): 
		
		# Get edges as list of tuples of gene names and obtain subgraph
		edges = [ self.edges[j] for j in self.tree_edges[i] ]
		H = self.G.edge_subgraph(edges).copy()
		
		_,_,normalized_components = self.output_normalized_decomposition()
		nx.set_node_attributes(H, {gene:normalized_components.values[i][j] for j,gene in enumerate(self.nodes)}, 'loadings')
		return H
	
	def get_component_sizes(self): 
		return (self.components != 0).sum(axis=1)

	def get_supports_as_genes(self): 
		return [ self._to_genes(np.where(component != 0)[0]) for component in self.components ]
	
	def get_max_degrees(self): 
#		 trees = [self.get_component_as_networkx(i) for i in range(self.K)]
#		 return [ max(dict(H.degree()).values()) for H in trees ]
		return np.array([max(np.append(np.unique(self._edges_pcst[self.tree_edges[i]], return_counts=True)[1], 0)) for i in range(self.K)])
	
	def output_support_jaccard(self): 
		S = self.get_supports_as_genes()
		jaccard = [[len(set(S[a]) & set(S[b])) / len(set(S[a]) | set(S[b])) for b in range(self.K)] for a in range(self.K)]
		return jaccard
	
	def output_scores_as_df(self): 
		return pd.DataFrame(self.scores, index=self.X_samples)
	
	def output_components_as_df(self): 
		return pd.DataFrame(self.components, columns=self.X_genes)
	
	
	def output_normalized_decomposition(self): 
		scale_components = (self.components ** 2).sum(axis=1) ** 0.5
		normalized_components = self.components * np.expand_dims(1/scale_components, axis=1)
		normalized_components = pd.DataFrame(normalized_components, columns=self.X_genes)
		
		scale_scores = (self.scores ** 2).sum(axis=0) ** 0.5
		normalized_scores = self.scores * np.expand_dims(1/scale_scores, axis=0)
		normalized_scores = pd.DataFrame(normalized_scores, self.X_samples)
		
		singular_values = scale_components * scale_scores
		
		# X ~ (normalized_scores * singular_values).dot(normalized_components)
		
		return normalized_scores, singular_values, normalized_components