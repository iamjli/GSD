#!/usr/bin/env python3

# Data processing
import pandas as pd
import numpy as np

from sklearn.decomposition import FastICA
from scipy import stats



def ICA1(X, cutoff, n_components): 
	
	ica = FastICA(n_components=n_components, max_iter=1000)
	# Because n_samples < n_features, we must transpose
	sources = ica.fit_transform(X.T).T  # (n_sources, n_features)
	mixing = ica.mixing_  # (n_samples, n_sources)
	
	out = []
	for source in sources: 
		# Z-score normalize
		z_scores = stats.zscore(source)
		# Get p-value from area under the tail for each z score
		p_values = stats.norm.sf(abs(z_scores))*2
		# Select top genes according to threshold
		support  = (p_values < cutoff).astype(int) 
		out.append(support)
		
	return np.array(out)


def ICA2(X, cutoff, n_components): 
	
	# Same as in ICA1, but here we find the support of significant positive and negative values separately
	ica = FastICA(n_components=n_components, max_iter=1000)
	# Because n_samples < n_features, we must transpose
	sources = ica.fit_transform(X.T).T  # (n_sources, n_features)
	mixing = ica.mixing_  # (n_samples, n_sources)
	
	out = []
	for source in sources: 
		# Z-score normalize
		z_scores = stats.zscore(source)
		# Positive support
		pos_p_values = stats.norm.sf(np.clip(z_scores, a_min=0, a_max=None)) * 2
		pos_support  = (pos_p_values < cutoff).astype(int) 
		out.append(pos_support)
		# Negative support
		neg_p_values = stats.norm.sf(np.clip(z_scores*-1, a_min=0, a_max=None)) * 2
		neg_support  = (neg_p_values < cutoff).astype(int) 
		out.append(neg_support)
		
	return np.array(out)