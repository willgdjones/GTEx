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
sys.path.insert(0, os.getcwd())
from src.utils.helpers import *


GTEx_directory = '/hps/nobackup/research/stegle/users/willj/GTEx'

parser = argparse.ArgumentParser(description='Collection of experiments. Runs on the cluster.')
parser.add_argument('-g', '--group', help='Experiment group', required=True)
parser.add_argument('-n', '--name', help='Experiment name', required=True)
args = vars(parser.parse_args())
group = args['group']
name = args['name']


class NIPSQuestion1():

    @staticmethod
    def calculate_variance_explained():

        os.makedirs(GTEx_directory + '/results/{}'.format(group), exist_ok=True)

        TISSUES = ['Lung', 'Artery - Tibial', 'Heart - Left Ventricle', 'Breast - Mammary Tissue', 'Brain - Cerebellum', 'Pancreas', 'Testis', 'Liver', 'Ovary', 'Stomach']
        SIZES = ['128', '256', '512', '1024', '2048', '4096']
        AGGREGATIONS = ['mean', 'median']
        MODELS = ['raw', 'retrained']

        results = {}

        for t in TISSUES:
            for a in AGGREGATIONS:
                for m in MODELS:
                    for s in SIZES:
                        key = '{}_{}_{}_{}'.format(t,a,m,s)

                        print(key)
                        image_features, expression, donorIDs, transcriptIDs, technical_factors, technical_headers, technical_idx = extract_final_layer_data(t, m, a, s)

                        print ("Calculating PCs that explain 99.9% of image feature variance")
                        pca_im = PCA(n_components=0.999)
                        pca_image_features = pca_im.fit_transform(image_features)


                        print ("Calculating PCs that explain 99.9% of expression variance")
                        pca_exp = PCA(n_components=0.999)
                        pca_expression = pca_exp.fit_transform(expression)

                        print ("Computing correlation matrix")
                        N = pca_expression.shape[1]
                        M = pca_image_features.shape[1]
                        R_matrix = np.zeros(shape=(N,M))
                        for i in range(N):
                            for j in range(M):
                                R, pv = pearsonr(pca_expression[:,i], pca_image_features[:,j])
                                R_matrix[i,j] = R

                        print ("Calculating variance explained")
                        variance_explained = pca_im.explained_variance_
                        total = sum([variance_explained[k] * sum(R_matrix[k,:]**2) for k in range(len(variance_explained))])
                        print (total)

                        results[key] = total


        pickle.dump(results, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))




class Question2():
    @staticmethod
    def shared_variability():

        TISSUES = ['Lung', 'Artery - Tibial', 'Heart - Left Ventricle', 'Breast - Mammary Tissue', 'Brain - Cerebellum', 'Pancreas', 'Testis', 'Liver', 'Ovary', 'Stomach']
        SIZES = ['128', '256', '512', '1024', '2048', '4096']
        AGGREGATIONS = ['mean', 'median']
        MODELS = ['raw', 'retrained']
        TFs = ['SMTSISCH', 'SMNTRNRT', 'SMEXNCRT', 'SMRIN', 'SMATSSCR']

        results = {}

        for t in TISSUES:
            for a in AGGREGATIONS:
                for m in MODELS:
                    for s in SIZES:
                        key = '{}_{}_{}_{}'.format(t,a,m,s)

                        Y, X, dIDs, tIDs, tfs, ths, t_idx = extract_final_layer_data(t, m, a, s)
                        Y_prime = Y[t_idx,:]
                        X_prime = X[t_idx,:]
                        tf_idx = [list(ths).index(x) for x in TFs]
                        tf_X = tfs[:,tf_idx]
                        # Take log of SMTSISH
                        tf_X[tf_idx.index('SMTSISCH')] = np.log2(tf_idx.index('SMTSISCH'))
                        lr_X = LinearRegression()
                        lr_Y = LinearRegression()
                        lr.fit(tf_X, Y_prime)
                        predicted = lr.predict(tf_X)
                        corrected_Y = Y_prime - predicted
                        all_Y.append(corrected_Y)







        Y_prime = Y[t_idx,:]
        X_prime = X[t_idx,:]
        TFs = ['SMTSISCH', 'SMNTRNRT', 'SMEXNCRT', 'SMRIN', 'SMATSSCR']
        tf_idx = [list(ths).index(x) for x in TFs]
        tf_X = tfs[:,tf_idx]
        lr = LinearRegression()
        lr.fit(tf_X, Y_prime)
        predicted = lr.predict(tf_X)
        corrected_Y = Y_prime - predicted
        all_Y.append(corrected_Y)









if __name__ == '__main__':
    eval(group + '().' + name + '()')
