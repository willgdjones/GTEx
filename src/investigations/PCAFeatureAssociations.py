import os
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import h5py
import argparse
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from matplotlib.colors import Normalize

from utils.helpers import *


GTEx_directory = '/hps/nobackup/research/stegle/users/willj/GTEx'

parser = argparse.ArgumentParser(description='Collection of experiments. Runs on the cluster.')
parser.add_argument('-g', '--group', help='Experiment group', required=True)
parser.add_argument('-n', '--name', help='Experiment name', required=True)
args = vars(parser.parse_args())
group = args['group']
name = args['name']

class PCAFeatureAssociations():

    @staticmethod
    def expression_PCs_vs_TFs():
        os.makedirs(GTEx_directory + '/results/{}'.format(group), exist_ok=True)

        features, expression, donorIDs, transcriptIDs, technical_factors, technical_headers, technical_idx = extract_final_layer_data('Lung', 'retrained', 'mean', '256')

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

        image_features, expression, donorIDs, transcriptIDs, technical_factors, technical_headers, technical_idx = extract_final_layer_data('Lung', 'retrained', 'mean', '256')

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


    @staticmethod
    def expression_PCs_vs_image_feature_PCs():

        image_features, expression, donorIDs, transcriptIDs, technical_factors, technical_headers, technical_idx = extract_final_layer_data('Lung', 'retrained', 'mean', '256')

        os.makedirs(GTEx_directory + '/results/{}'.format(group), exist_ok=True)


        print ("Calculating PCs that explain 95% of image feature variance")
        pca = PCA(n_components=0.95)
        pca_image_features = pca.fit_transform(image_features)


        print ("Calculating PCs that explain 95% of expression variance")
        pca = PCA(n_components=0.95)
        pca_expression = pca.fit_transform(expression)



        N = pca_expression.shape[1]
        M = pca_image_features.shape[1]
        R_matrix = np.zeros(shape=(N,M))
        pv_matrix = np.zeros(shape=(N,M))
        for i in range(N):
            for j in range(M):
                R, pv = pearsonr(pca_expression[:,i], pca_image_features[:,j])
                R_matrix[i,j] = R
                pv_matrix[i,j] = pv

        print ("Extracting data for the top 5 associations")
        top5_scatter_results = {}
        sorted_idx = np.argsort((R_matrix**2).flatten())[::-1]
        top5_scatter_results['sorted_idx'] = sorted_idx

        for k in range(5):
            idx = sorted_idx[k]
            image_pc_idx = idx % M
            exp_pc_idx = int(idx / M)
            image_pc_vector = pca_image_features[:, image_pc_idx]
            expression_pc_vector = pca_expression[:, exp_pc_idx]
            image_feature_pc_number = image_pc_idx + 1
            expression_pc_number = exp_pc_idx + 1
            R = R_matrix.flatten()[idx]
            pv = pv_matrix.flatten()[idx]
            top5_scatter_results[k] = [expression_pc_number, image_feature_pc_number, expression_pc_vector, image_pc_vector, R, pv]

        results = [R_matrix, pv_matrix, top5_scatter_results]

        pickle.dump(results, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))




if __name__ == '__main__':
    eval(group + '().' + name + '()')
