#!/usr/bin/env python3

# Core python modules
import os
import logging
from functools import partial

# Python external libraries
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import FastICA

# Internal modules
from gsd import GSD



logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(asctime)s - Models: %(levelname)s - %(message)s', "%I:%M:%S"))
logger.addHandler(handler)


def save_results(X_path, out_path, model_params):

	if os.path.exists(out_path): 
		logger.info("Results have already been generated: {}".format(out_path))
		return 

	X = np.load(X_path)
	
	method = model_params.pop("method", None)
	if method == "ICA1": 
		results = ICA1(X, **model_params)

	elif method == "GSD": 
		results = run_GSD(X, **model_params)

	else:
		pass

	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	np.save(out_path, results)
	logger.info("Run complete, results written to: {}".format(out_path))


def get_model(method, **params): 

	if method == "ICA1": 
		func = partial(ICA1, **params)
	elif method == "GSD": 
		func = partial(run_GSD, **params)
	else: 
		logger.warn("No models found with method {}".format(method))
	return func


def get_model_tag(method, **params): 

	if method == "ICA1":
		tag = "ICA1_{n_components}_{cutoff}".format(**params)
	elif method == "GSD": 
		tag = "GSD_{a}_{n_components}_{initializer}".format(**params)
	else: 
		logger.warn("No models found with method {}".format(method))
	return tag


######## GSD ########

def run_GSD(X, n_iterations, **params): 
	
	gsd = GSD(X, **params)
	
	for _ in range(n_iterations): 
		for i in range(gsd.n_components): 
			gsd.update(i)
			
	return gsd.components


######## BENCHMARKS ########

def _significant_weights(component, cutoff, mode='all'):
	"""
	Sparsifies component, keeping most significant values. 
	"""

	z_scores = stats.zscore(component)

	# Get p-values for each z-score from a 2-tailed distribution
	if   mode == 'all':		 p_vals = stats.norm.sf(abs(z_scores)) * 2
	elif mode == 'positive': p_vals = stats.norm.sf(z_scores.clip(min=0)) * 2
	elif mode == 'negative': p_vals = stats.norm.sf((-z_scores).clip(min=0)) * 2
	else: pass

	# Set values that dont pass the threshold to 0
	out = component.copy()
	out[~(p_vals < cutoff)] = 0
	return out


def ICA1(X, cutoff, n_components): 
	
	ica = FastICA(n_components=n_components, max_iter=1000)
	# Because n_samples < n_features, we must transpose.
	sources = ica.fit_transform(X.T).T  # (n_sources, n_features)
	mixing = ica.mixing_  # (n_samples, n_sources)
	
	out = [ _significant_weights(source, cutoff) for source in sources ]
	return np.array(out)


def ICA2(X, cutoff, n_components): 
	
	# Same as in ICA1, but here we find the support of significant positive and negative values separately
	ica = FastICA(n_components=n_components, max_iter=1000)
	# Because n_samples < n_features, we must transpose
	sources = ica.fit_transform(X.T).T  # (n_sources, n_features)
	mixing = ica.mixing_  # (n_samples, n_sources)
	
	pos_out = [ _significant_weights(source, cutoff, mode='positive') for source in sources ]
	neg_out = [ _significant_weights(source, cutoff, mode='negative') for source in sources ]
	out = pos_out + neg_out
	return np.array(out)