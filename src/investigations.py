import os
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import h5py
import argparse
from sklearn.decomposition import PCA
from matplotlib.colors import Normalize
from scipy.stats import pearsonr
from utils.helpers import *


GTEx_directory = '/hps/nobackup/research/stegle/users/willj/GTEx'

parser = argparse.ArgumentParser(description='Collection of experiments. Runs on the cluster.')
parser.add_argument('-g', '--group', help='Experiment group', required=True)
parser.add_argument('-n', '--name', help='Experiment name', required=True)
args = vars(parser.parse_args())
group = args['group']
name = args['name']

class Classifier():

    @staticmethod
    def validation_accuracy_across_patchsize():
        os.makedirs(GTEx_directory + '/results/{}'.format(group), exist_ok=True)
        validation_accuracies = []
        for k in range(1, 7):
            histories = pickle.load(open(GTEx_directory + '/models/histories_50_-{}.py'.format(k),'rb'))
            validation_accuracies.append(histories[1]['val_acc'][-1])
        np.savetxt(GTEx_directory + '/results/{group}/{name}.txt'.format(group=group, name=name), validation_accuracies)


class PCAFeatureAssociations():

    @staticmethod
    def expression_PCs_vs_TFs():
        os.makedirs(GTEx_directory + '/results/{}'.format(group), exist_ok=True)

        features, expression, donorIDs, transcriptIDs, technical_factors, technical_headers, technical_idx = extract_final_layer_data('Lung', 'retrained', 'median', '256')

        print ("Calculating PCs that explain 95% of variance")
        pca = PCA(n_components=0.95)
        pca_expression = pca.fit_transform(expression)
        pca_expression = pca_expression[technical_idx,:]

        N = technical_factors.shape[1]
        M = pca_expression.shape[1]
        R_matrix = np.zeros(shape=(N,M))
        pv_matrix = np.zeros(shape=(N,M))
        for i in range(N):
            for j in range(M):
                R, pv = pearsonr(technical_factors[:,i], pca_expression[:,j])
                R_matrix[i,j] = R
                pv_matrix[i,j] = pv

        print ("Extracting data for the top 5 associations")
        top5_scatter_results = {}
        sorted_idx = np.argsort((R_matrix**2).flatten())[::-1]
        top5_scatter_results['sorted_idx'] = sorted_idx

        for k in range(5):
            idx = sorted_idx[k]
            pc_idx = idx % M
            tf_idx = int(idx / M)
            pc_vector = pca_expression[:,pc_idx]
            tf_vector = technical_factors[:,tf_idx]
            tf_names = technical_headers[tf_idx]
            pc_number = pc_idx + 1
            R = R_matrix.flatten()[idx]
            pv = pv_matrix.flatten()[idx]
            top5_scatter_results[k] = [tf_names, pc_number, tf_vector, pc_vector, R, pv]

        results = [R_matrix, pv_matrix, top5_scatter_results]

        pickle.dump(results, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))

    @staticmethod
    def image_feature_PCs_vs_TFs():
        os.makedirs(GTEx_directory + '/results/{}'.format(group), exist_ok=True)

        image_features, expression, donorIDs, transcriptIDs, technical_factors, technical_headers, technical_idx = extract_final_layer_data('Lung', 'retrained', 'median', '256')

        print ("Calculating PCs that explain 95% of variance")
        pca = PCA(n_components=0.95)
        pca_image_features = pca.fit_transform(image_features)
        pca_image_features= pca_image_features[technical_idx,:]

        N = technical_factors.shape[1]
        M = pca_image_features.shape[1]
        R_matrix = np.zeros(shape=(N,M))
        pv_matrix = np.zeros(shape=(N,M))
        for i in range(N):
            for j in range(M):
                R, pv = pearsonr(technical_factors[:,i], pca_image_features[:,j])
                R_matrix[i,j] = R
                pv_matrix[i,j] = pv

        print ("Extracting data for the top 5 associations")
        top5_scatter_results = {}
        sorted_idx = np.argsort((R_matrix**2).flatten())[::-1]
        top5_scatter_results['sorted_idx'] = sorted_idx

        for k in range(5):
            idx = sorted_idx[k]
            pc_idx = idx % M
            tf_idx = int(idx / M)
            pc_vector = pca_image_features[:,pc_idx]
            tf_vector = technical_factors[:,tf_idx]
            tf_names = technical_headers[tf_idx]
            pc_number = pc_idx + 1
            R = R_matrix.flatten()[idx]
            pv = pv_matrix.flatten()[idx]
            top5_scatter_results[k] = [tf_names, pc_number, tf_vector, pc_vector, R, pv]

        results = [R_matrix, pv_matrix, top5_scatter_results]

        pickle.dump(results, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))










#
#
# retrained_mean_features = {}
# with h5py.File(GTEx_directory + '/small_data/new_retrained_inceptionet_aggregations.hdf5','r') as f:
# 	expression = f['lung']['256']['expression'].value
# 	for s in ['128','256','512','1024','2048','4096']:
# 		size_retrained_mean_features = f['lung'][s]['mean'].value
# 		retrained_mean_features[s] = size_retrained_mean_features
#
# 	expression_IDs = f['lung']['256']['expression_IDs'].value
#
# raw_mean_features = {}
# with h5py.File(GTEx_directory + '/small_data/new_raw_inceptionet_aggregations.hdf5','r') as f:
# 	for s in ['128','256','512','1024','2048','4096']:
# 		size_raw_mean_features = f['lung'][s]['mean'].value
# 		size_raw_mean_features[size_raw_mean_features < 0] = 0
# 		raw_mean_features[s] = size_raw_mean_features
#
# most_expressed_transcript_idx, retrained_most_varying_feature_idx, retrained_results = pickle.load(open(GTEx_directory + '/small_data/retrained_quick_pvalues.py','rb'))
# most_expressed_transcript_idx, raw_most_varying_feature_idx, raw_results = pickle.load(open(GTEx_directory + '/small_data/raw_quick_pvalues.py','rb'))
#
#
#
#
#
# GTEx_directory = '/hps/nobackup/research/stegle/users/willj/GTEx'
#
# with h5py.File(GTEx_directory + '/small_data/new_retrained_inceptionet_aggregations.hdf5', 'r') as f:
#     expression = f['lung']['256']['expression'].value
#     transcriptIDs = f['lung']['256']['expression_IDs'].value
#     features = f['lung']['256']['mean'].value
#     donorIDs = f['lung']['256']['donor_IDs'].value
#     donorIDs = [x.decode('utf-8').split('-')[1] for x in donorIDs]
#     donorIDs = [x.encode('ascii') for x in donorIDs]
#     technical_factors, technical_headers, technical_idx = get_technical_factors('Lung', donorIDs)
#
#
# from sklearn.decomposition import PCA
# pca = PCA(n_components=50)
# expression_pca = pca.fit_transform(expression)
#
# filt_features, most_varying_feature_idx = filter_features(features, 500)
# filt_expression, filt_transcriptIDs, transcript_idx = filter_expression(expression, transcriptIDs, 1000)

if __name__ == '__main__':
    eval(group + '().' + name + '()')
