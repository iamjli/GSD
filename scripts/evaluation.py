#!/usr/bin/env python3

# Core python modules
import os
import multiprocessing
import logging

from itertools import product

import numpy as np
from sklearn.model_selection import ParameterGrid

from scoring import recovery_relevance, precision_recall



logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(asctime)s - Evaluate: %(levelname)s - %(message)s', "%I:%M:%S"))
logger.addHandler(handler)


# Count the number of available CPUs for potential use in multiprocessing code
try: n_cpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
except KeyError: n_cpus = multiprocessing.cpu_count()


class Evaluate: 

	def __init__(self, graph, sampler, param_grid=dict()): 

		self.graph = graph
		self.sampler = sampler
		self.param_grid  = param_grid

		# Ensure param_grid has the necessary attributes
		for attribute in [ "n_sources", "n_samples", "support_size", "noise", "rep" ]: 
			assert attribute in param_grid

		# Create lookup tables
		self._matrix_lookups()
		self.data_params = ParameterGrid(self.param_grid)


	########  DATA GENERATION  ######## 

	def _param_iterator(self, keys): 
		# Returns an iterator through parameter sets with specified keys
		return product(*[ self.param_grid[key] for key in keys ])

	def _source_matrix(self, support_size, n_sources):
		# Matrix of genes x sources (aka components)
		# This should be shared across all parameter sets with the same `n_sources` and `support_size_vals`
		sources = np.array([ self.sampler(self.graph, support_size) for _ in range(n_sources) ]).T  
		return sources

	def _loadings_matrix(self, n_sources, n_samples):
		# Matrix of sources x sample latent values 
		loadings = np.random.rand(n_sources, n_samples)
		return loadings

	def _data_matrix(self, support_size, n_sources, n_samples, noise, rep): 
		# Load corresponding source and loading matrices
		sources = self.get_sources(support_size, n_sources, rep)
		loadings = self.get_loadings(n_sources, n_samples, rep)
		# Reconstruct data matrix by taking dot product and adding noise
		data = np.random.normal( sources.dot(loadings), scale=noise)
		return data

	def _matrix_lookups(self): 

		# Dictionary of source signals indexed by (n_sources, support_size)
		source_params = self._param_iterator(['support_size', 'n_sources', 'rep'])
		self.source_matrices = { (support_size, n_sources, rep) : self._source_matrix(support_size, n_sources) for support_size,n_sources,rep in source_params }

		# Dictionary of loadings indexed by (n_sources, n_samples)
		loading_params = self._param_iterator(['n_sources', 'n_samples', 'rep'])
		self.loading_matrices = { (n_sources, n_samples, rep) : self._loadings_matrix(n_sources, n_samples) for n_sources,n_samples,rep in loading_params }

		# Dictionary of data matrices
		data_params = self._param_iterator(['support_size', 'n_sources', 'n_samples', 'noise', 'rep'])
		self.data_matrices = { params : self._data_matrix(*params) for params in data_params }


	########  EVALUATION  ######## 

	def evaluate(self, model): 

		with multiprocessing.Pool(n_cpus) as pool: 

			data = [ self.get_data(**params) for params in self.data_params ]
			results = pool.map(model, data)

			comparisons = [ (result, self.get_sources(**params).T) for result,params in zip(*[results, self.data_params]) ]
			scores = pool.starmap(recovery_relevance, comparisons)

		return scores


	########  DATA QUERY  ######## 

	def get_sources(self, support_size, n_sources, rep, **kwargs): 
		return self.source_matrices[(support_size, n_sources, rep)]

	def get_loadings(self, n_sources, n_samples, rep, **kwargs): 
		return self.loading_matrices[(n_sources, n_samples, rep)]

	def get_data(self, support_size, n_sources, n_samples, noise, rep, **kwargs): 
		return self.data_matrices[(support_size, n_sources, n_samples, noise, rep)]


