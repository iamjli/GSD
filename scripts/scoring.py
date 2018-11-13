#!/usr/bin/env python3

import numpy as np
from scipy import stats


# These functions are used to score model results against known results


def recovery_relevance(model, known): 

	# If signal value is non-zero, assign to 1
	supp1 = (model != 0).astype(float)
	supp2 = (known != 0).astype(float) 

	# For each pair of modules, find number of nodes in intersect and union
	# Dot product of two supports gives number of nodes in intersection
	intersect = supp1.dot(supp2.T)
	union = supp1.shape[1] - (1-supp1).dot(1-supp2.T) 
	jaccard_matrix = intersect / union

	recovery  = jaccard_matrix.max(axis=0).mean()
	relevance = jaccard_matrix.max(axis=1).mean()
	if recovery * relevance == 0: F1rr = 0
	else: F1rr = stats.hmean([recovery, relevance])

	scores = {
		"recovery": recovery, 
		"relevance": relevance, 
		"F1rr": F1rr
	}

	return scores


def precision_recall(model, known): 

	## WARNING !! 
	## This function needs to be verified for accuracy. 

	## TODO: Could potentially be optimized by removing all nan values first.

	# If signal value is non-zero, assign to 1
	supp1 = (model != 0).astype(float)
	supp2 = (known != 0).astype(float) 

	# For recall and precision, we need to find number of times pairwise elements appear in known and model modules
	# Given a boolean module vector, element (i,j) is 1 if and only ifboth i and j are in the module 
	# To count the number of modules that each pair is in, we need to perform an outer product for each module
	# This can be efficiently computed with a matrix multiplication
	pairwise1 = np.matmul(supp1.T, supp1)
	pairwise2 = np.matmul(supp2.T, supp2)
	min_pairwise = np.minimum(pairwise1,pairwise2)

	pairwise1[pairwise1 == 0] = np.nan
	pairwise2[pairwise2 == 0] = np.nan
	pairwise_norm1 = np.divide(min_pairwise, pairwise1)
	pairwise_norm2 = np.divide(min_pairwise, pairwise2)

	# We need to take the mean of the lower triangular matrix excluding the diagonal, so we set the upper 
	# triangular matrix to nan, then take the mean. 
	pairwise_norm1[np.triu_indices(pairwise_norm1.shape[0])] = np.nan
	pairwise_norm2[np.triu_indices(pairwise_norm2.shape[0])] = np.nan

	# We can run into problems if the overlap here is sparse
	if np.count_nonzero(~np.isnan(pairwise_norm1)) == 0: precision = 0
	else: precision = np.nanmean(pairwise_norm1)

	if np.count_nonzero(~np.isnan(pairwise_norm2)) == 0: recall = 0
	else: recall = np.nanmean(pairwise_norm2)

	if recall * precision == 0: F1rp = 0
	else: F1rp = stats.hmean([recall, precision])

	scores = {
		"precision": precision, 
		"recall": recall, 
		"F1rp": F1rp
	}

	return scores