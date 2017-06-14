import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import h5py
from scipy.stats import pearsonr

GTEx_directory = '/Users/fonz/Documents/Projects/GTEx'

# [expression, expression_IDs, retrained_mean_features] = pickle.load(open(GTEx_directory + '/small_data/lung_mean_256_and_expression.py','rb'))
retrained_mean_features = {}
with h5py.File(GTEx_directory + '/small_data/new_retrained_inceptionet_aggregations.hdf5','r') as f:
	expression = f['lung']['256']['expression'].value
	for s in ['128','256','512','1024','2048','4096']:
		size_retrained_mean_features = f['lung'][s]['mean'].value
		retrained_mean_features[s] = size_retrained_mean_features

	expression_IDs = f['lung']['256']['expression_IDs'].value

raw_mean_features = {}
with h5py.File(GTEx_directory + '/small_data/new_raw_inceptionet_aggregations.hdf5','r') as f:
	for s in ['128','256','512','1024','2048','4096']:
		size_raw_mean_features = f['lung'][s]['mean'].value
		size_raw_mean_features[size_raw_mean_features < 0] = 0
		raw_mean_features[s] = size_raw_mean_features



# import pdb; pdb.set_trace()
raw_concatenated_features = np.vstack([raw_mean_features['128'], raw_mean_features['256'], raw_mean_features['512'], raw_mean_features['1024'], raw_mean_features['2048'], raw_mean_features['4096']])
retrained_concatenated_features = np.vstack([retrained_mean_features['128'], retrained_mean_features['256'], retrained_mean_features['512'], retrained_mean_features['1024'], retrained_mean_features['2048'], retrained_mean_features['4096']])
most_expressed_transcript_idx = list(np.argsort(np.std(expression,axis=0))[-1000:]) + list(np.argsort(np.mean(expression,axis=0))[-1000:])
raw_most_varying_feature_idx = np.argsort(np.std(raw_concatenated_features,axis=0))[::-1][:500]
retrained_most_varying_feature_idx = np.argsort(np.std(retrained_concatenated_features,axis=0))[::-1][:500]

raw_results = {}
aggregations = ['mean','median']
sizes = ['128','256','512','1024','2048','4096']
for a in aggregations:
	for s in sizes:
		print (a,s)
		with h5py.File(GTEx_directory + '/small_data/new_raw_inceptionet_aggregations.hdf5','r') as f:
			features = f['lung'][s][a].value
			features[features < 0] = 0

			filt_expression = expression[:,most_expressed_transcript_idx]
			filt_features = features[:,raw_most_varying_feature_idx]
			N = 500
			M = 2000
			pvalues = np.zeros((N,M))
			R_matrix = np.zeros((N,M))
			for i in range(N):
				if i%10 == 0:
					print (i)
				for j in range(M):
					res = pearsonr(filt_expression[:,j], filt_features[:,i])
					R_matrix[i,j] = res[0]
					pvalues[i,j] = res[1]

			R_key = '{}_{}_{}'.format(a,s,'R')
			pvalue_key = '{}_{}_{}'.format(a,s,'pvalues')
			raw_results[R_key] = R_matrix
			raw_results[pvalue_key] = pvalues


pickle.dump([most_expressed_transcript_idx, raw_most_varying_feature_idx, raw_results], open(GTEx_directory + '/small_data/raw_pvalues.py','wb'))


retrained_results = {}
aggregations = ['mean','median']
sizes = ['128','256','512','1024','2048','4096']
for a in aggregations:
	for s in sizes:
		print (a,s)
		with h5py.File(GTEx_directory + '/small_data/new_retrained_inceptionet_aggregations.hdf5','r') as f:
			features = f['lung'][s][a].value

			filt_expression = expression[:,most_expressed_transcript_idx]
			filt_features = features[:,retrained_most_varying_feature_idx]

			N = 500
			M = 2000
			pvalues = np.zeros((N,M))
			R_matrix = np.zeros((N,M))
			for i in range(N):
				if i%10 == 0:
					print (i)
				for j in range(M):
					res = pearsonr(filt_expression[:,j], filt_features[:,i])
					R_matrix[i,j] = res[0]
					pvalues[i,j] = res[1]

			R_key = '{}_{}_{}'.format(a,s,'R')
			pvalue_key = '{}_{}_{}'.format(a,s,'pvalues')
			retrained_results[R_key] = R_matrix
			retrained_results[pvalue_key] = pvalues


import pickle; pickle.dump([most_expressed_transcript_idx, retrained_most_varying_feature_idx, retrained_results], open(GTEx_directory + '/small_data/retrained_pvalues.py','wb'))
