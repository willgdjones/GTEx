import pickle
import numpy as np
import argparse
# from sklearn.decomposition import PCA
# from sklearn.linear_model import LinearRegression
# from matplotlib.colors import Normalize
import os
import sys
sys.path.insert(0, os.getcwd())
from src.utils.helpers import *


GTEx_directory = '/hps/nobackup/research/stegle/users/willj/GTEx'

parser = argparse.ArgumentParser(description='Collection of experiments. Runs on the cluster.')
parser.add_argument('-g', '--group', help='Experiment group', required=True)
parser.add_argument('-n', '--name', help='Experiment name', required=True)
args = vars(parser.parse_args())
group = args['group']
name = args['name']

class InflationPvalues():

    @staticmethod
    def raw_pvalues():
        os.makedirs(GTEx_directory + '/results/{}'.format(group), exist_ok=True)

        association_results, most_varying_feature_idx, filt_transcriptIDs = pickle.load(open(GTEx_directory + '/intermediate_results/RawFeatureAssociations/raw_pvalues.pickle', 'rb'))
        results = association_results['Lung_mean_retrained_256']

        pickle.dump(results, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))

    @staticmethod
    def pc_corrected_pvalues():
        os.makedirs(GTEx_directory + '/results/{}'.format(group), exist_ok=True)
        association_results, most_varying_feature_idx, filt_transcriptIDs = pickle.load(open(GTEx_directory + '/intermediate_results/PCCorrectedFeatureAssociations/corrected_pvalues.pickle', 'rb'))
        results = [association_results['Lung_mean_retrained_256_{}'.format(i)] for i in [1,2,3,4,5]]


        pickle.dump(results, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))

    @staticmethod
    def tf_corrected_pvalues():
        os.makedirs(GTEx_directory + '/results/{}'.format(group), exist_ok=True)
        association_results, most_varying_feature_idx, filt_transcriptIDs = pickle.load(open(GTEx_directory + '/intermediate_results/TFCorrectedFeatureAssociations/corrected_pvalues.pickle', 'rb'))
        results = association_results['Lung_mean_retrained_256']


        pickle.dump(results, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))



if __name__ == '__main__':
    eval(group + '().' + name + '()')
