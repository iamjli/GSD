

"""

This file contains different methods to sample source signals from a protein network (interactome). 


Eventually may be best implemented as a class: 

Args: 
	graph: networkx object
	kwargs: attributes like signal size

Functions: 
	Samplers: 
		test
		...

	Return sampler functions potentially with arguments partially filled

"""


import numpy as np
import pandas as pd


def gene_list_to_boolean_signal(graph, gene_list): 
	# Each gene in gene_list is assigned a value 1, the rest are assigned 0
	signal_series = pd.Series(data=1, index=gene_list).reindex(graph.nodes).fillna(0).astype(int)
	return signal_series.values


########  SAMPLERS  ########

def sample_test(graph, size): 

	# Sample nodes without replacement
	random_nodes = np.random.choice(graph.nodes, size, replace=False) 

	return gene_list_to_boolean_signal(graph, random_nodes)


def sample_random_walk_fixed_size(graph, size): 

	# Beginning with a random node, randomly sample until the number of visited nodes is reaches specified
	visited_nodes = list(np.random.choice(graph, 1))
	current_node = visited_nodes[0]
	while len(visited_nodes) < size: 
		genes,scores = zip(*[[neighbor,cofidence] for _,neighbor,cofidence in graph.edges(current_node, data="confidence")])
		genes = list(genes)
		scores = np.array(scores) / sum(scores)
		
		current_node = np.random.choice(genes,1,p=scores)[0]
		if current_node not in visited_nodes: 
			visited_nodes.append(current_node)

	return gene_list_to_boolean_signal(graph, visited_nodes)


def sample_random_walk_fixed_iters(graph, iters):

	# Beginning with a random node, sample according to random walk and report nodes visited after a specified number of iterations
	visited_nodes = list(np.random.choice(graph, 1))
	current_node = visited_nodes[0]
	
	for _ in range(iters): 
		genes,scores = zip(*[[neighbor,cofidence] for _,neighbor,cofidence in graph.edges(current_node, data="confidence")])
		genes = list(genes)
		scores = np.array(scores) / sum(scores)
		
		current_node = np.random.choice(genes,1,p=scores)[0]
		if current_node not in visited_nodes: 
			visited_nodes.append(current_node)

	return gene_list_to_boolean_signal(graph, visited_nodes)
