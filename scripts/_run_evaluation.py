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
from _samplers import SourceSampler, LoadingSampler
from _models import get_model, get_model_tags, save_model

from scoring import recovery_relevance, precision_recall


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(asctime)s - Data: %(levelname)s - %(message)s', "%I:%M:%S"))
logger.addHandler(handler)


n_cpus = multiprocessing.cpu_count()


def initialize_data(data_params):

	source_template	  = os.path.join(home_dir, "data", "source", source_tag)
	loading_template	 = os.path.join(home_dir, "data", "loading", loading_tag)
	data_matrix_template = os.path.join(home_dir, "data", "data_matrix", data_matrix_tag)

	os.makedirs(os.path.dirname(source_template), exist_ok=True)
	os.makedirs(os.path.dirname(loading_template), exist_ok=True)
	os.makedirs(os.path.dirname(data_matrix_template), exist_ok=True)

	# Import samplers
	source_sampler  = SourceSampler(data_params['graph'], data_params['source_sampler_method'])
	loading_sampler = LoadingSampler(data_params['loading_sampler_method'])

	for param in ParameterGrid(data_params['grid']):
		
		# Load source matrix, or generate and save if file doesn't exist yet.
		source_path = source_template.format(**param)
		if os.path.exists(source_path): 
			sources = np.load(source_path)
		else: 
			sources = source_sampler.sample(**param)
			np.save(source_path, sources)
			
		
		# Load loading matrix, or generate and save if file doesn't exist yet.
		loading_path = loading_template.format(**param)
		if os.path.exists(loading_path): 
			loadings = np.load(loading_path)
		else: 
			loadings = loading_sampler.sample(**param)
			np.save(loading_path, loadings) 
		
		
		# Generate and save data matrix files
		data_matrix_path = data_matrix_template.format(**param)
		if os.path.exists(data_matrix_path): 
			data_matrix = np.load(data_matrix_path)
		else: 
			data_matrix = np.random.normal( loadings.dot(sources), scale=param['noise'] ** 0.5)
			np.save(data_matrix_path, data_matrix)
			logger.info("Wrote data matrix to {}".format(data_matrix_path))



source_tag = "{n_sources}_{size}_{rep}.npy"
loading_tag = "{n_samples}_{n_sources}_{rep}.npy"
data_matrix_tag = "{n_samples}_{n_sources}_{noise}_{size}_{rep}.npy"

home_dir = "../_evaluation"


def main():

	data_param_file = os.path.join(home_dir, "data_parameters.json")
	model_param_file = os.path.join(home_dir, "model_parameters.json")

	with open(data_param_file) as f:  data_params = json.load(f)
	with open(model_param_file) as f: models_params = json.load(f)

	initialize_data(data_params)

	# Generator of model tag, model pairs
	model_generator = ([get_model_tags(model_attr['method'], **params), get_model(model_attr['method'], **params)] 
		for model_attr in models_params for params in ParameterGrid(model_attr['param_grid']))

	# Run parameter against all models
	for tag, model in model_generator: 

		saver = save_results(model=model, tag=tag, home_dir=home_dir)

		with multiprocessing.Pool(n_cpus) as pool: 

			pool.map(saver, ParameterGrid(data_params['grid']))





# source_rel_path = os.path.join("data", "sources", "{n_sources}_{size}_{rep}.npy")
# loading_rel_path = os.path.join("data", "loading", "{n_samples}_{n_sources}_{rep}.npy")
# data_matrix_rel_path = os.path.join("data", "data_matrix", "{n_samples}_{n_sources}_{noise}_{size}_{rep}.npy")




def save_results(model, tag, home_dir): 
	return partial(save_model_results, model=model, tag=tag, home_dir=home_dir)


def save_model_results(data_params, model, tag, home_dir): 

	path = os.path.join(home_dir, tag, "{n_samples}_{n_sources}_{noise}_{size}_{rep}.npy".format(**data_params))
	os.makedirs(os.path.dirname(path), exist_ok=True)

	print(path)

	X = np.load(os.path.join(home_dir, "data", "data_matrix", data_matrix_tag.format(**data_params)))
	results = model(X)
	np.save(path, results)

	logger.info("Wrote to {}".format(path))



if __name__ == "__main__":
	main()

	# home_dir = "../_evaluation"

	# data_param_file = os.path.join(home_dir, "data_parameters.json")
	# model_param_file = os.path.join(home_dir, "model_parameters.json")

	# with open(data_param_file) as f:  data_params = json.load(f)
	# with open(model_param_file) as f: models_params = json.load(f)


	# initialize_data(home_dir, data_params)

	# # Generator of model tag, model pairs
	# model_generator = ([get_model_tags(model_attr['method'], **params), get_model(model_attr['method'], **params)] 
	# 	for model_attr in models_params for params in ParameterGrid(model_attr['param_grid']))

	# # Run parameter against all models
	# for tag, model in model_generator: 

	# 	saver = save_results(model=model, tag=tag, home_dir=home_dir)

	# 	with multiprocessing.Pool(n_cpus) as pool: 

	# 		pool.map(saver, ParameterGrid(data_params['grid']))

	# 	# for param in ParameterGrid(data_params['grid']):
	# 	# 	print(param)
	# 	# 	X = np.load(os.path.join(home_dir, data_matrix_rel_path.format(**param)))
	# 	# 	save_results(X, data_params=param)







	# for mp in models_params: 
	# 	for model_param in ParameterGrid(mp['param_grid']): 
	# 		model = get_model(mp['method'], **model_param)
	# 		X_paths = [os.path.join(home_dir, data_matrix_rel_path).format(**param) for param in ParameterGrid(data_params['grid'])]

	# 		results = evaluate_model(model, X_paths)



	# evaluate_model(home_dir, models_params, data_params)



