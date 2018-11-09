



"""

This file contains methods to evaluate different models performance on synthetic and known data. 


 * signal sampler (pure source sources)
 	* n_sources
 	* signal support size
 	* [ sampler kernel ]
 * signal reconstruction
 	* n_samples
 	* noise
 	* [ mixing proportions ]

 * model
 	* model
 	* model param


Variable scheme

Data generation
	Signal
	Mixing
	Noisy data 
	(data not needed because can be generated easily)

Separately: n_samples. Remove this from grid. 



0) For each param except for `n_samples` and `noise`, 

1) Sample `n_sources` source signal(s) from the signal generator into a 2D numpy array, 
   where each column is a signal component and each row is a gene. Set as param attribute

2) Initialize loadings (coeffs corresponding to contribution of each component) as a 2D np array
   where each column is the latent representation for a single sample, and rows correspond to sources.

3) Reconstruct data by dot product and add `noise` 


"""

import numpy as np

from sklearn.model_selection import ParameterGrid




def subset_dict(dictionary, keys): return dict((key, dictionary[key]) for key in dictionary.keys() & set(keys))


class Evaluate: 

	def __init__(self, graph, sampler, param_grid=dict()): 

		self.graph = graph
		self.sampler = sampler
		self.param_grid  = param_grid

		# Ensure param_grid has the necessary attributes
		assert 'n_sources' in param_grid
		assert 'n_samples' in param_grid
		assert 'support_size' in param_grid
		assert 'noise' in param_grid

		print("Parameter grid loaded:", subset_dict(self.param_grid, ['n_sources', 'support_size']))

		# Set lookup dictionaries to sample from for different parameter sets
		self.source_matrices, self.loading_matrices = self._matrix_lookups()

		# Generate data matrices for each parameter set in the same order as 
		self.params_iterator = ParameterGrid(self.param_grid)
		self.data_matrices = [ self._data_matrix(**params) for params in self.params_iterator ]


	def evaluate(self, ): 
		pass


	def _source_matrix(self, support_size, n_sources):

		# Matrix of genes x sources (aka components)
		# This should be shared across all parameter sets with the same `n_sources` and `support_size_vals`
		sources = np.array([ self.sampler(support_size) for _ in range(n_sources) ]).T  
		return sources

	def _loadings_matrix(self, n_sources, n_samples):

		# Matrix of sources x sample latent values 
		loadings = np.random.rand(n_sources, n_samples)
		return loadings

	def _matrix_lookups(self): 

		# Dictionary of source signals indexed by (n_sources, support_size)
		source_param_grid = ParameterGrid( subset_dict(self.param_grid, ['support_size', 'n_sources']) )
		source_matrices = { (params['support_size'], params['n_sources']) : self._source_matrix(**params) for params in source_param_grid }

		# Dictionary of loadings indexed by (n_sources, n_samples)
		loading_param_grid = ParameterGrid( subset_dict(self.param_grid, ['n_sources', 'n_samples']) )
		loading_matrices = { (params['n_sources'], params['n_samples']) : self._loadings_matrix(**params) for params in loading_param_grid }

		return source_matrices, loading_matrices


	def _data_matrix(self, support_size, n_sources, n_samples, noise): 

		sources = self.source_matrices[(support_size, n_sources)]
		loadings = self.loading_matrices[(n_sources, n_samples)]

		# Add noise to the reconstructed and add to params dict
		# Matrix of genes x samples
		data = np.random.normal( sources.dot(loadings), scale=noise)
		return data








# class Evaluate:

# 	def __init__(self, graph, sampler, param_grid): 

# 		self.graph = graph
# 		self.sampler = sampler
# 		self.param_grid = param_grid

# 		assert "n_samples" in self.param_grid
# 		# and so on

# 		self.max_n_samples = max(self.param_grid['n_samples'])

# 		# Build lookup dict to map parameters to sampled data
# 		self.data_matrix_lookup = self._build_data_matrix_lookup()


# 	def _build_data_matrix_lookup(self):

# 		data_matrix_lookup = {}

# 		# Retrieve parameter values necessary for generating the data matrix
# 		data_param_grid = { param : self.param_grid[param] for param in ['n_sources', 'support_size', 'noise'] }

# 		for params in ParameterGrid(data_param_grid): 

# 			# Add data matrices keyed by parameter
# 			params_key = (params['n_sources'], params['support_size'], params['noise'])
# 			data_matrix_lookup[params_key] = self._sample_data(**params)

# 		return data_matrix_lookup


# 	def _sample_data(self, n_sources, support_size, noise): 

# 		# Matrix of genes x sources (aka components)
# 		sources = np.array([ self.sampler(support_size) for _ in range(n_sources) ]).T  
# 		# Matrix of sources x sample latent values 
# 		loadings = np.random.rand(n_sources, self.max_n_samples)

# 		# Add noise to the reconstructed and add to params dict
# 		# Matrix of genes x samples
# 		data = np.random.normal( sources.dot(loadings), scale=noise)

# 		matrices = dict(source_matrix=sources, loading_matrix=loadings, data_matrix=data)

# 		return matrices


# 	def query_data_matrix(self, n_sources, support_size, noise, n_samples): 

# 		# Get data matrix from lookup dictionary
# 		params_key = (n_sources, support_size, noise)
# 		full_data_matrix = self.data_matrix_lookup[params_key]['data_matrix']
# 		# Return the correct number of samples
# 		data_matrix = full_data_matrix[:,:n_samples]

# 		return data_matrix



