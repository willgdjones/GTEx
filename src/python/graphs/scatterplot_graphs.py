import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import h5py


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

most_expressed_transcript_idx, retrained_most_varying_feature_idx, retrained_results = pickle.load(open(GTEx_directory + '/small_data/retrained_quick_pvalues.py','rb'))
most_expressed_transcript_idx, raw_most_varying_feature_idx, raw_results = pickle.load(open(GTEx_directory + '/small_data/raw_quick_pvalues.py','rb'))
