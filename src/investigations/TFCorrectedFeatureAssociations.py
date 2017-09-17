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


class TFCorrectedFeatureAssociations():

    @staticmethod
    def corrected_pvalues():

        os.makedirs(GTEx_directory + '/intermediate_results/{}'.format(group), exist_ok=True)
        SIZES = [128, 256, 512, 1024, 2048, 4096]
        AGGREGATIONS = ['mean', 'median']
        MODELS = ['raw', 'retrained']

        N = 500
        M = 2000
        k = 1

        association_results = {}
        most_varying_feature_indexes = {}

        print('Computing associations between {} transcripts and {} features'.format(M,N))

        for a in AGGREGATIONS:
            for m in MODELS:

                print ('Filtering features for Lung {} {}'.format(m, a))
                all_filt_features, most_varying_feature_idx, expression, _, transcriptIDs, tfs, ths, t_idx = filter_features_across_all_patchsizes('Lung', m, a, N, tf_correction=True)
                filt_expression, filt_transcriptIDs = filter_expression(expression, transcriptIDs, M, k)
                for s in SIZES:
                    key = '{}_{}_{}_{}'.format('Lung', a, m, s)
                    filt_features = all_filt_features[s]

                    print('Computing: Lung', a, m, s)
                    filt_features_copy = filt_features.copy()
                    filt_expression_prime = filt_expression[t_idx,:]
                    res = compute_pearsonR(filt_features_copy, filt_expression_prime)

                    most_varying_feature_indexes[key] = most_varying_feature_idx
                    association_results[key] = res


        results = [association_results, most_varying_feature_indexes, filt_transcriptIDs]

        pickle.dump(results, open(GTEx_directory + '/intermediate_results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))

if __name__ == '__main__':
    eval(group + '().' + name + '()')
