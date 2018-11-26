#!/usr/bin/env python3

# Core python modules
import sys, os
import json
import logging
import multiprocessing
from functools import partial

# Python external libraries
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import ParameterGrid

# Peripheral python modules
import argparse

# Internal modules
from samplers import SourceSampler, LoadingSampler
from models import save_results, get_model_tag
from scoring import recovery_relevance, precision_recall


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(asctime)s - Evaluate: %(levelname)s - %(message)s', "%I:%M:%S"))
logger.addHandler(handler)


n_cpus = multiprocessing.cpu_count()

class Evaluate:

	def __init__(self, home_dir, data_specs_file, model_specs_file): 

		self.home_dir = home_dir
		self.data_specs_file = data_specs_file
		self.model_specs_file = model_specs_file

		# Directory names for source, loading, and data matrix files.
		self.D_dir = os.path.join(self.home_dir, "data", "sources")
		self.Z_dir = os.path.join(self.home_dir, "data", "loadings")
		self.X_dir = os.path.join(self.home_dir, "data", "data_matrices")

		# Load data and model parameter specifications
		with open(self.data_specs_file) as f:  self.data_specs  = json.load(f)
		with open(self.model_specs_file) as f: self.model_specs = json.load(f)

		# Iterators for data and model parameters
		self.data_paramlist = ParameterGrid(self.data_specs['grid'])
		# For each model method, create parameter dict from each parameter grid.
		self.model_paramlist = [ { **p, 'method':model_attr['method'] } for model_attr in self.model_specs for p in ParameterGrid(model_attr['grid']) ]

		# Sampler objects for generating source and loading matrices as well as their tags.
		self.source_sampler  = SourceSampler(self.data_specs['graph'], self.data_specs['source_sampler_method'])
		self.loading_sampler = LoadingSampler(self.data_specs['loading_sampler_method'])

		# Get path from parameters
		self._data_tag = lambda param: "D_{}_Z_{}_noise_{}.npy".format(self.source_sampler.tag(**param), self.loading_sampler.tag(**param), param['noise'])
		self._D_path   = lambda param: os.path.join(self.D_dir, self.source_sampler.tag(**param) + ".npy")
		self._Z_path   = lambda param: os.path.join(self.Z_dir, self.loading_sampler.tag(**param) + ".npy")
		self._X_path   = lambda param: os.path.join(self.X_dir, self._data_tag(param))


	def run_grid(self, model_param): 

		logger.info("Running {} with parameters: {}".format(model_param["method"], model_param))

		results_dir = os.path.join(self.home_dir, "results", get_model_tag(**model_param))
		os.makedirs(results_dir, exist_ok=True)
		io_paths = [ (self._X_path(p), os.path.join(results_dir, self._data_tag(p))) for p in self.data_paramlist ]

		with multiprocessing.Pool(n_cpus) as pool: 
			pool.starmap(partial(save_results, **model_param), io_paths)


	def score_results(self, model_param): 

		model_tag = get_model_tag(**model_param)
		results_dir = os.path.join(self.home_dir, "results", model_tag)
		results_paths = [ (os.path.join(results_dir, self._data_tag(p)), self._D_path(p)) for p in self.data_paramlist ]

		# Populates a list with scores in the same order as `self.data_paramlist`.
		with multiprocessing.Pool(n_cpus) as pool: 
			scores = pool.starmap(recovery_relevance, results_paths)

		scores = [ {**score, **param, "model":model_tag} for score,param in zip(*[ scores, self.data_paramlist ]) ]

		return scores 


	def get_scores_as_dataframe(self): 

		scores = []

		for model_params in self.model_paramlist: 
			scores += self.score_results(model_params)
			
		scores_df = pd.DataFrame(scores)
		
		return scores_df


	def initialize_data(self):

		os.makedirs(self.D_dir, exist_ok=True)
		os.makedirs(self.Z_dir, exist_ok=True)
		os.makedirs(self.X_dir, exist_ok=True)

		for param in ParameterGrid(self.data_specs['grid']): 

			D_path = self._D_path(param)
			Z_path = self._Z_path(param)
			X_path = self._X_path(param)

			# Load source matrix, or generate and save if file doesn't exist yet.
			if os.path.exists(D_path):
				D = np.load(D_path)
			else: 
				D = self.source_sampler.sample(**param)
				np.save(D_path, D)

			# Load loading matrix, or generate and save if file doesn't exist yet.
			if os.path.exists(Z_path):
				Z = np.load(Z_path)
			else: 
				Z = self.loading_sampler.sample(**param)
				np.save(Z_path, Z)

			# Generate and save data matrix files.
			if os.path.exists(X_path):
				X = np.load(X_path)
				logger.info("Data matrix exists: {}.".format(X_path))
			else: 
				X = np.random.normal( Z.dot(D), scale=param['noise'] ** 0.5)
				np.save(X_path, X)
				logger.info("Wrote data matrix: {}.".format(X_path))



parser = argparse.ArgumentParser(description="""Evaluate different models for synthetic data.""")

class FullPaths(argparse.Action):
	"""Expand user- and relative-paths"""
	def __call__(self,parser, namespace, values, option_string=None):
		setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))

def directory(dirname):
	if not os.path.isdir(dirname): raise argparse.ArgumentTypeError(dirname + " is not a directory")
	else: return dirname

# Input / Output parameters:
io_params = parser.add_argument_group("Input / Output Files")

io_params.add_argument("-o", "--output", dest='output_dir', action=FullPaths, type=directory, required=True,
	help='(Required) Output directory path')
io_params.add_argument("-d", "--data_specs", dest='data_specs_file', type=str, required=False,
	help='Path to data specifications file')
io_params.add_argument("-m", "--model_specs", dest='model_specs_file', type=str, required=False,
	help='Path to model specifications file')


def main(): 

	args = parser.parse_args()

	output_dir = args.output_dir
	data_specs_file = args.data_specs_file
	model_specs_file = args.model_specs_file

	if data_specs_file is None: data_specs_file = os.path.join(output_dir, "data_specs.json")
	if model_specs_file is None: model_specs_file = os.path.join(output_dir, "model_specs.json")

	eval = Evaluate(output_dir, data_specs_file, model_specs_file)

	eval.initialize_data()

	for model_params in eval.model_paramlist: 
		eval.run_grid(model_params)


if __name__ == "__main__": 
	main()