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

class RawFeatureAssociations():

    @staticmethod
    def compute_pvalues():
        os.makedirs(GTEx_directory + '/intermediate_results/{}'.format(group), exist_ok=True)
        SIZES = ['128', '256', '512', '1024', '2048', '4096']
        AGGREGATIONS = ['mean', 'median']
        MODELS = ['raw', 'retrained']
        M = 2000
        k = 1

        association_results = {}
        most_varying_feature_indexes = {}

        print('Computing raw associations')

        for a in AGGREGATIONS:
            for m in MODELS:
                for s in SIZES:
                    key = '{}_{}_{}_{}'.format('Lung', a, m, s)
                    Y, X, dIDs, filt_tIDs, tfs, ths, t_idx = filter_and_correct_expression_and_image_features('Lung', m, a, s, M, k, pc_correction=False, tf_correction=False)
                    N = Y.shape[1]
                    print('Computing {} x {} = {} associations for: Lung'.format(N, M, N*M), a, m, s)
                    res = compute_pearsonR(Y, X)
                    association_results[key] = res

        results = [association_results, most_varying_feature_indexes, filt_transcriptIDs]

        pickle.dump(results, open(GTEx_directory + '/intermediate_results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))

    @staticmethod
    def raw_associations_across_patchsizes():

        import statsmodels.stats.multitest as smm
        os.makedirs(GTEx_directory + '/results/{}'.format(group), exist_ok=True)

        association_results, most_varying_feature_idx, filt_transcriptIDs = pickle.load(open(GTEx_directory + '/intermediate_results/RawFeatureAssociations/raw_pvalues.pickle', 'rb'))
        SIZES = [128, 256, 512, 1024, 2048, 4096]
        ALPHAS = [0.01,0.0001,0.000001]

        print ("Calculating Bonferroni significant associations:")
        all_counts = []
        for alph in ALPHAS:
            print ("Alpha: ", alph)
            size_counts = []
            for s in SIZES:
                print ("Patch size: ", s)
                pvalues = association_results['{}_{}_{}_{}'.format('Lung','median','retrained',s)][1].flatten()
                counts = sum(smm.multipletests(pvalues, method='bonferroni',alpha=alph)[0])
                size_counts.append(counts)
            all_counts.append(size_counts)

        print ("Saving results")
        pickle.dump(all_counts, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))

    @staticmethod
    def associations_raw_vs_retrained():

        import statsmodels.stats.multitest as smm
        os.makedirs(GTEx_directory + '/results/{}'.format(group), exist_ok=True)

        association_results, most_varying_feature_idx, filt_transcriptIDs = pickle.load(open(GTEx_directory + '/intermediate_results/RawFeatureAssociations/raw_pvalues.pickle', 'rb'))

        alpha = 0.0001
        SIZES = [128, 256, 512, 1024, 2048, 4096]
        MODELS = ['retrained', 'raw']

        print ("Calculating Bonferroni significant associations:")
        all_counts = []
        for m in MODELS:
            print ("Model: ", m)
            model_counts = []
            for s in SIZES:
                print ("Patch size: ", s)
                pvalues = association_results['{}_{}_{}_{}'.format('Lung','median',m,s)][1].flatten()
                counts = sum(smm.multipletests(pvalues, method='bonferroni',alpha=alpha)[0])
                model_counts.append(counts)
            all_counts.append(model_counts)

        print ("Saving results")
        pickle.dump(all_counts, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))


    @staticmethod
    def associations_mean_vs_median():

        import statsmodels.stats.multitest as smm

        os.makedirs(GTEx_directory + '/results/{}'.format(group), exist_ok=True)
        association_results, most_varying_feature_idx, filt_transcriptIDs = pickle.load(open(GTEx_directory + '/intermediate_results/RawFeatureAssociations/raw_pvalues.pickle', 'rb'))

        alpha = 0.0001
        SIZES = [128, 256, 512, 1024, 2048, 4096]
        AGGREGATIONS = ['mean', 'median']

        print ("Calculating Bonferroni significant associations:")
        all_counts = []
        for a in AGGREGATIONS:
            print ("Aggregation: ", a)
            aggregation_counts = []
            for s in SIZES:
                print ("Patch size: ", s)
                pvalues = association_results['{}_{}_{}_{}'.format('Lung',a,'retrained',s)][1].flatten()
                counts = sum(smm.multipletests(pvalues, method='bonferroni',alpha=alpha)[0])
                aggregation_counts.append(counts)
            all_counts.append(aggregation_counts)

        print ("Saving results")
        pickle.dump(all_counts, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))


    @staticmethod
    def features_with_significant_transcripts():

        import statsmodels.stats.multitest as smm

        os.makedirs(GTEx_directory + '/results/{}'.format(group), exist_ok=True)
        association_results, most_varying_feature_idx, filt_transcriptIDs = pickle.load(open(GTEx_directory + '/intermediate_results/RawFeatureAssociations/raw_pvalues.pickle', 'rb'))

        alpha = 0.0001
        SIZES = [128, 256, 512, 1024, 2048, 4096]


        print ("Calculating Bonferroni significant associations:")
        size_counts = []
        for s in SIZES:
            print ("Patch size: ", s)
            pvalues = association_results['{}_{}_{}_{}'.format('Lung','mean','retrained',s)][1]
            original_shape = pvalues.shape
            counts = sum(np.sum(smm.multipletests(pvalues.flatten(),method='bonferroni',alpha=alpha)[0].reshape(original_shape),axis=1) > 0)
            size_counts.append(counts)

        print ("Saving results")
        pickle.dump(size_counts, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))


    @staticmethod
    def transcripts_with_significant_features():

        import statsmodels.stats.multitest as smm

        os.makedirs(GTEx_directory + '/results/{}'.format(group), exist_ok=True)
        association_results, most_varying_feature_idx, filt_transcriptIDs = pickle.load(open(GTEx_directory + '/intermediate_results/RawFeatureAssociations/raw_pvalues.pickle', 'rb'))

        alpha = 0.0001
        SIZES = [128, 256, 512, 1024, 2048, 4096]


        print ("Calculating Bonferroni significant associations:")
        size_counts = []
        for s in SIZES:
            print ("Patch size: ", s)
            pvalues = association_results['{}_{}_{}_{}'.format('Lung','mean','retrained',s)][1]
            original_shape = pvalues.shape
            counts = sum(np.sum(smm.multipletests(pvalues.flatten(),method='bonferroni',alpha=alpha)[0].reshape(original_shape),axis=0) > 0)
            size_counts.append(counts)

        print ("Saving results")
        pickle.dump(size_counts, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))


    @staticmethod
    def image_feature_796_vs_SMTSISCH():
        image_features, expression, donorIDs, transcriptIDs, technical_factors, technical_headers, technical_idx = extract_final_layer_data('Lung', 'retrained', 'mean', '256')


        filt_features = image_features[technical_idx,:]
        SMTSISCH = np.squeeze(technical_factors[:,technical_headers == 'SMTSISCH'])
        feature796 =filt_features[:,795]

        results = [SMTSISCH, feature796]

        pickle.dump(results, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))


    @staticmethod
    def image_feature_671_vs_SMTSISCH():
        image_features, expression, donorIDs, transcriptIDs, technical_factors, technical_headers, technical_idx = extract_final_layer_data('Lung', 'retrained', 'mean', '256')


        filt_features = image_features[technical_idx,:]
        SMTSISCH = np.squeeze(technical_factors[:,technical_headers == 'SMTSISCH'])
        feature671 =filt_features[:,670]

        results = [SMTSISCH, feature671]

        pickle.dump(results, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))

    @staticmethod
    def image_feature_671_vs_TFs():
        image_features, expression, donorIDs, transcriptIDs, technical_factors, technical_headers, technical_idx = extract_final_layer_data('Lung', 'retrained', 'mean', '256')
        filt_features = image_features[technical_idx,:]
        SMTSISCH = np.squeeze(technical_factors[:,technical_headers == 'SMTSISCH'])
        SMRIN = np.squeeze(technical_factors[:,technical_headers == 'SMRIN'])
        SMEXNCRT = np.squeeze(technical_factors[:,technical_headers == 'SMEXNCRT'])
        SMNTRNRT = np.squeeze(technical_factors[:,technical_headers == 'SMNTRNRT'])
        SMATSSCR = np.squeeze(technical_factors[:,technical_headers == 'SMATSSCR'])
        feature671 =filt_features[:,670]

        results = [SMATSSCR, SMNTRNRT, SMTSISCH, SMEXNCRT, feature671, SMRIN]

        pickle.dump(results, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))


    @staticmethod
    def image_feature_796_vs_TFs():
        image_features, expression, donorIDs, transcriptIDs, technical_factors, technical_headers, technical_idx = extract_final_layer_data('Lung', 'retrained', 'mean', '256')
        filt_features = image_features[technical_idx,:]
        SMTSISCH = np.squeeze(technical_factors[:,technical_headers == 'SMTSISCH'])
        SMRIN = np.squeeze(technical_factors[:,technical_headers == 'SMRIN'])
        SMEXNCRT = np.squeeze(technical_factors[:,technical_headers == 'SMEXNCRT'])
        SMNTRNRT = np.squeeze(technical_factors[:,technical_headers == 'SMNTRNRT'])
        SMATSSCR = np.squeeze(technical_factors[:,technical_headers == 'SMATSSCR'])
        feature671 =filt_features[:,795]

        results = [SMATSSCR, SMNTRNRT, SMTSISCH, SMEXNCRT, feature671, SMRIN]

        pickle.dump(results, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))




if __name__ == '__main__':
    eval(group + '().' + name + '()')
