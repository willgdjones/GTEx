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
import statsmodels.stats.multitest as smm
from gprofiler import GProfiler
from pyensembl import EnsemblRelease
data = EnsemblRelease(77)
import multiprocess as mp
from tqdm import tqdm
import time
from pebble import ProcessPool, ProcessExpired
from concurrent.futures import TimeoutError

GTEx_directory = '/hps/nobackup/research/stegle/users/willj/GTEx'

parser = argparse.ArgumentParser(description='Collection of experiments. Runs on the cluster.')
parser.add_argument('-g', '--group', help='Experiment group', required=True)
parser.add_argument('-n', '--name', help='Experiment name', required=True)
parser.add_argument('-p', '--params', help='Parameters')
args = vars(parser.parse_args())
group = args['group']
name = args['name']
parameter_key = args['params']

class FeatureSelection():

    @staticmethod
    def tf_feature_selection_expression():
        Y, X, dIDs, tIDs, tfs, ths, t_idx = extract_final_layer_data('Lung', 'retrained', 'mean', '256')


        tf_Y = Y[t_idx,:]
        tf_X = X[t_idx,:]

        tfs[:,list(ths).index('SMTSISCH')] = np.log2(tfs[:,list(ths).index('SMTSISCH')] + 1)


        print ("Calculating PCs that explain 99.9% of image feature variance")
        PCA_X = PCA(n_components=0.999)
        pca_X = PCA_X.fit_transform(tf_X)

        idx = np.zeros_like(ths)
        N = pca_X.shape[1]
        ordered_choices = [(0, None)]

        print ("Performing feature selection")

        for i in range(51):
            print ("Iteration {}".format(i))
        #
            selected_c = np.argwhere(idx == 1).flatten()
            selected_predictors = tfs[:,selected_c]
        #     lr = LinearRegression()
        #     pca_Y_copy = pca_Y.copy()
        #     if i > 0:
        #         lr.fit(tf_predictors, pca_Y)
        #         regressed_pca_Y = lr.predict(tf_predictors)
        #         regressed_pca_Y = pca_Y_copy - regressed_pca_Y
        #     else:
        #         regressed_pca_Y = pca_Y_copy

            max_frac_var_explained, max_choice = [0, None]
            unselected_c = np.argwhere(idx == 0).flatten()
            print ("{} choices left".format(len(unselected_c)))

            for choice in unselected_c:

                if i == 0:
                    trial_predictors = tfs[:, choice].reshape(-1,1)
                else:

                    trial_predictors = np.zeros((tfs.shape[0], len(selected_c) + 1))
                    trial_predictors[:,:-1] = selected_predictors
                    trial_predictors[:,-1] = tfs[:, choice]
                lr = LinearRegression()

                lr.fit(trial_predictors, pca_X)
                residuals = pca_X - lr.predict(trial_predictors)
                var_explained_per_PC = 1 - (np.var(residuals, axis=0) / np.var(pca_X, axis=0))


                frac_var_explained = np.dot(PCA_X.explained_variance_, var_explained_per_PC) / sum(PCA_X.explained_variance_)


                if frac_var_explained > max_frac_var_explained:
                    max_frac_var_explained, max_choice = frac_var_explained, choice



            print ((max_frac_var_explained, max_choice))
            print ("With {}, can explain {} of variance".format(ths[max_choice], max_frac_var_explained))
            ordered_choices.append((max_frac_var_explained, ths[max_choice]))
            idx[max_choice] = 1

        pickle.dump(ordered_choices, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))


    @staticmethod
    def tf_feature_selection_image_features():
        Y, X, dIDs, tIDs, tfs, ths, t_idx = extract_final_layer_data('Lung', 'retrained', 'mean', '256')


        tf_Y = Y[t_idx,:]
        tf_X = X[t_idx,:]

        tfs[:,list(ths).index('SMTSISCH')] = np.log2(tfs[:,list(ths).index('SMTSISCH')] + 1)


        print ("Calculating PCs that explain 99.9% of image feature variance")
        PCA_Y = PCA(n_components=0.999)
        pca_Y = PCA_Y.fit_transform(tf_Y)

        idx = np.zeros_like(ths)
        N = pca_Y.shape[1]
        ordered_choices = [(0, None)]

        print ("Performing feature selection")

        extra_frac_variance_explained = np.inf

        for i in range(51):
            print ("Iteration {}".format(i))
        #
            selected_c = np.argwhere(idx == 1).flatten()
            selected_predictors = tfs[:,selected_c]
        #     lr = LinearRegression()
        #     pca_Y_copy = pca_Y.copy()
        #     if i > 0:
        #         lr.fit(tf_predictors, pca_Y)
        #         regressed_pca_Y = lr.predict(tf_predictors)
        #         regressed_pca_Y = pca_Y_copy - regressed_pca_Y
        #     else:
        #         regressed_pca_Y = pca_Y_copy

            max_frac_var_explained, max_choice = [0, None]
            unselected_c = np.argwhere(idx == 0).flatten()
            print ("{} choices left".format(len(unselected_c)))

            variance_of_choices = []

            for choice in unselected_c:

                if i == 0:
                    trial_predictors = tfs[:, choice].reshape(-1,1)
                else:

                    trial_predictors = np.zeros((tfs.shape[0], len(selected_c) + 1))
                    trial_predictors[:,:-1] = selected_predictors
                    trial_predictors[:,-1] = tfs[:, choice]
                lr = LinearRegression()

                lr.fit(trial_predictors, pca_Y)
                residuals = pca_Y - lr.predict(trial_predictors)
                var_explained_per_PC = 1 - (np.var(residuals, axis=0) / np.var(pca_Y, axis=0))


                frac_var_explained = np.dot(PCA_Y.explained_variance_, var_explained_per_PC) / sum(PCA_Y.explained_variance_)

                variance_of_choices.append((choice, frac_var_explained))


                if frac_var_explained > max_frac_var_explained:
                    max_frac_var_explained, max_choice = frac_var_explained, choice

            max_idx = np.argmax([x[1] for x in variance_of_choices])


            max_choice = variance_of_choices[max_idx][0]
            max_frac_var_explained = variance_of_choices[max_idx][1]



            print ((max_frac_var_explained, max_choice))
            print ("With {}, can explain {} of variance".format(ths[max_choice], max_frac_var_explained))
            ordered_choices.append((max_frac_var_explained, ths[max_choice]))
            new_extra_frac_variance_explained = max_frac_var_explained - ordered_choices[i][0]

            
            print("Extra variance explained: {}".format(extra_frac_variance_explained))
            extra_frac_variance_explained = new_extra_frac_variance_explained
            idx[max_choice] = 1

        pickle.dump(ordered_choices, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))



if __name__ == '__main__':
    eval(group + '().' + name + '()')
