#!/usr/bin/env python3

# Data processing
import pandas as pd
import numpy as np

from sklearn.decomposition import FastICA
from scipy import stats


def _significant_weights(component, cutoff, mode='all'):
	"""
	Sparsifies component, keeping most significant values. 
	"""

	z_scores = stats.zscore(component)

	# Get p-values for each z-score from a 2-tailed distribution
	if   mode == 'all':      p_vals = stats.norm.sf(abs(z_scores)) * 2
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
