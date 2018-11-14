#!/usr/bin/env python3

# Core python modules
import os
import multiprocessing
import logging

from itertools import product

# Data manipulation modules
import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid

from scoring import recovery_relevance, precision_recall



logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(asctime)s - Data: %(levelname)s - %(message)s', "%I:%M:%S"))
logger.addHandler(handler)


# Count the number of available CPUs for potential use in multiprocessing code
try: n_cpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
except KeyError: n_cpus = multiprocessing.cpu_count()


class Data: 

	def __init__(self, sampler, param_grid=dict()): 

		self.sampler = sampler
		self.param_grid = param_grid

		# Ensure param_grid has the necessary attributes
		for attribute in [ "n_sources", "n_samples", "noise", "rep" ]: 
			assert attribute in self.param_grid

		# Create lookup tables
		self._set_lookup_matrices()
		self.paramlist = list(ParameterGrid(self.param_grid))
		logger.info("{} parameter sets loaded.".format(len(self.paramlist)))

	########  DATA GENERATION  ######## 

	def _param_iterator(self, keys): 
		# Returns an iterator through parameter sets with specified keys
		return product(*[ self.param_grid[key] for key in keys ])

	def _source_matrix(self, n_sources): 
		# Shape (n_sources, genes)
		return np.array([ self.sampler() for _ in range(n_sources) ])

	def _loadings_matrix(self, n_samples, n_sources):
		# Shape (n_samples, n_sources)
		return np.random.rand(n_samples, n_sources)

	def _data_matrix(self, n_sources, n_samples, noise, rep):
		# Load corresponding source and loading matrices
		sources = self.get_sources(n_sources, rep)
		loadings = self.get_loadings(n_samples, n_sources, rep)
		# Reconstruct data matrix by taking dot product and adding noise
		data = np.random.normal( loadings.dot(sources), scale=noise ** 0.5)
		return data

	def _set_lookup_matrices(self): 
		# Dictionary of source signals indexed by (n_sources, rep)
		source_params = self._param_iterator(['n_sources', 'rep'])
		self.source_matrices = { (n_sources, rep) : self._source_matrix(n_sources) for n_sources,rep in source_params }

		# Dictionary of loadings indexed by (n_samples, n_sources, rep)
		loading_params = self._param_iterator(['n_samples', 'n_sources', 'rep'])
		self.loading_matrices = { (n_samples, n_sources, rep) : self._loadings_matrix(n_samples, n_sources) for n_samples,n_sources,rep in loading_params }

		# Dictionary of data matrices
		data_params = self._param_iterator(['n_sources', 'n_samples', 'noise', 'rep'])
		self.data_matrices = { params : self._data_matrix(*params) for params in data_params }

	########  EVALUATION  ######## 

	def evaluate(self, model): 

		with multiprocessing.Pool(n_cpus) as pool: 
			
			data = [ self.get_data(**param) for param in self.paramlist ]
			results = pool.map(model, data)

			comparisons = [ (result, self.get_sources(**params)) for result,params in zip(*[results, self.paramlist]) ]
			scores = pool.starmap(recovery_relevance, comparisons)

		return results, scores


	def summarize_scores(self, scores): 

		params_df = pd.DataFrame(self.paramlist)
		scores_df = pd.DataFrame(scores)

		df = pd.concat([params_df, scores_df], axis=1)
		df = df.groupby(by=['n_samples', 'n_sources', 'noise']).agg([np.mean, np.std]).drop(columns=['rep'])
		
		return df

	########  DATA QUERY  ######## 

	def get_sources(self, n_sources, rep, **kwargs): 
		return self.source_matrices[(n_sources, rep)]

	def get_loadings(self, n_samples, n_sources, rep, **kwargs): 
		return self.loading_matrices[(n_samples, n_sources, rep)]

	def get_data(self, n_sources, n_samples, noise, rep, **kwargs): 
		return self.data_matrices[(n_sources, n_samples, noise, rep)]


