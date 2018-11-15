#!/usr/bin/env python3

# Core python modules
import sys, os
import json
import logging
import multiprocessing
from functools import partial

# Data processing
import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid

# Networks
import networkx as nx

# Modules
from samplers import SourceSampler, LoadingSampler
from models import save_results, get_model, get_model_tag

from scoring import recovery_relevance, precision_recall


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(asctime)s - Evaluate: %(levelname)s - %(message)s', "%I:%M:%S"))
logger.addHandler(handler)


n_cpus = multiprocessing.cpu_count()


class Evaluate:

	def __init__(self, home_dir, data_param_path, model_param_path, params=dict()): 

		self.home_dir = home_dir
		self.data_param_path = data_param_path
		self.model_param_path = model_param_path

		self.D_dir = os.path.join(self.home_dir, "data", "sources")
		self.Z_dir = os.path.join(self.home_dir, "data", "loadings")
		self.X_dir = os.path.join(self.home_dir, "data", "data_matrices")

		os.makedirs(self.D_dir, exist_ok=True)
		os.makedirs(self.Z_dir, exist_ok=True)
		os.makedirs(self.X_dir, exist_ok=True)


		with open(self.data_param_path) as f:  self.data_params = json.load(f)
		with open(self.model_param_path) as f: self.models_params = json.load(f)

		self.data_paramlist = ParameterGrid(self.data_params['grid'])
		self.model_paramlist = [ {**p, "method":model_attr['method']} for model_attr in self.models_params for p in ParameterGrid(model_attr['grid']) ]

		# self.initialize_data()


		# Loop through all models, run and evaluate and save results for each param

	def _data_tag(self, param): 
		return "{n_samples}_{n_sources}_{noise}_{size}_{rep}.npy".format(**param)

	def _D_path(self, param): 
		return os.path.join(self.D_dir, "{n_sources}_{size}_{rep}.npy".format(**param))

	def _Z_path(self, param):
		return os.path.join(self.Z_dir, "{n_samples}_{n_sources}_{rep}.npy".format(**param))

	def _X_path(self, param): 
		return os.path.join(self.X_dir, self._data_tag(param))


	def run_grid(self, model_param): 

		results_dir = os.path.join(self.home_dir, "results", get_model_tag(**model_param))
		os.makedirs(results_dir, exist_ok=True)
		io_paths = [ (self._X_path(p), os.path.join(results_dir, self._data_tag(p))) for p in self.data_paramlist]

		with multiprocessing.Pool(n_cpus) as pool: 
			pool.starmap(partial(save_results, model_params=model_param), io_paths)


	def initialize_data(self):

		source_sampler  = SourceSampler(self.data_params['graph'], self.data_params['source_sampler_method'])
		loading_sampler = LoadingSampler(self.data_params['loading_sampler_method'])

		for param in ParameterGrid(self.data_params['grid']): 

			D_path = self._D_path(param)
			Z_path = self._Z_path(param)
			X_path = self._X_path(param)

			# if ~os.path.exists(D_path): D = source_sampler.sample(**param)

			# Load source matrix, or generate and save if file doesn't exist yet.
			if os.path.exists(D_path):
				D = np.load(D_path)
			else: 
				D = source_sampler.sample(**param)
				np.save(D_path, D)

			# Load loading matrix, or generate and save if file doesn't exist yet.
			if os.path.exists(Z_path):
				Z = np.load(Z_path)
			else: 
				Z = loading_sampler.sample(**param)
				np.save(Z_path, Z)

			# Generate and save data matrix files.
			if os.path.exists(X_path):
				X = np.load(X_path)
				logger.info("Data matrix exists: {}.".format(X_path))
			else: 
				X = np.random.normal( Z.dot(D), scale=param['noise'] ** 0.5)
				np.save(X_path, X)
				logger.info("Wrote data matrix: {}.".format(X_path))
