#!/usr/bin/env python3

"""
This file contains different methods to sample signals from a protein network (interactome) as well as 
loadings matrices. 

"""

import numpy as np
import pandas as pd
import networkx as nx


class SourceSampler: 

	def __init__(self, graph, method): 

		# Import as networkx object if path is provided
		if isinstance(graph, str): self.graph = nx.read_gpickle(graph)
		else: self.graph = graph

		self.method = method


	def sample(self, n_sources, **params): 

		if self.method == 'random_walk_fixed_size': 
			source_matrix = np.array([ self._random_walk_fixed_size(size=params['size']) for _ in range(n_sources) ])

		return source_matrix


	########  SAMPLERS  ########

	def _gene_list_to_boolean_signal(self, gene_list): 
		# Each gene in gene_list is assigned a value 1, the rest are assigned 0
		signal_series = pd.Series(data=1, index=gene_list).reindex(self.graph.nodes).fillna(0).astype(int)
		return signal_series.values

	def _sample_test(self, size): 
		# Sample nodes without replacement
		random_nodes = np.random.choice(self.graph.nodes, size, replace=False) 

		return self._gene_list_to_boolean_signal(random_nodes)

	def _random_walk_fixed_size(self, size): 

		# Beginning with a random node, randomly sample until the number of visited nodes is reaches specified
		visited_nodes = list(np.random.choice(self.graph, 1))
		current_node = visited_nodes[0]
		while len(visited_nodes) < size: 
			genes,scores = zip(*[[neighbor,cofidence] for _,neighbor,cofidence in self.graph.edges(current_node, data="confidence")])
			genes = list(genes)
			scores = np.array(scores) / sum(scores)
			
			current_node = np.random.choice(genes,1,p=scores)[0]
			if current_node not in visited_nodes: 
				visited_nodes.append(current_node)

		return self._gene_list_to_boolean_signal(visited_nodes)

	def _random_walk_fixed_iters(self, iters):

		# Beginning with a random node, sample according to random walk and report nodes visited after a specified number of iterations
		visited_nodes = list(np.random.choice(self.graph, 1))
		current_node = visited_nodes[0]
		
		for _ in range(iters): 
			genes,scores = zip(*[[neighbor,confidence] for _,neighbor,confidence in self.graph.edges(current_node, data="confidence")])
			genes = list(genes)
			scores = np.array(scores) / sum(scores)
			
			current_node = np.random.choice(genes,1,p=scores)[0]
			if current_node not in visited_nodes: 
				visited_nodes.append(current_node)

		return self._gene_list_to_boolean_signal(visited_nodes)



class LoadingSampler: 

	def __init__(self, method): 

		self.method = method

	def sample(self, n_samples, n_sources, **kwargs): 

		if self.method == 'uniform': 

			return np.random.rand(n_samples, n_sources)

