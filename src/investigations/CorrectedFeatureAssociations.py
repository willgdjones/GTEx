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

class CorrectedFeatureAssociations():

    @staticmethod
    def corrected_pvalues():

        os.makedirs(GTEx_directory + '/intermediate_results/{}'.format(group), exist_ok=True)
        SIZES = [128, 256, 512, 1024, 2048, 4096]
        AGGREGATIONS = ['mean', 'median']
        MODELS = ['raw', 'retrained']
        PCs = [1,2,3]
        N = 500
        M = 2000
        k = 1

        association_results = {}

        for a in AGGREGATIONS:
            for m in MODELS:
                for pc in PCs:
                    print ('Filtering features for Lung {} {}'.format(m, a))
                    all_filt_features, most_varying_feature_idx, expression, _, transcriptIDs, _, _, _ = filter_features_across_all_patchsizes('Lung', m, a, N, pc_correction=pc)
                    filt_expression, filt_transcriptIDs = filter_expression(expression, transcriptIDs, M, k)
                    for s in SIZES:
                        filt_features = all_filt_features[s]
                        print('Computing: Lung', a, m, s, pc)
                        res = compute_pearsonR(filt_features, filt_expression)

                        association_results['{}_{}_{}_{}_{}'.format('Lung', a, m, s, pc)] = res

        results = [association_results, most_varying_feature_idx, filt_transcriptIDs]

        pickle.dump(results, open(GTEx_directory + '/intermediate_results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))


    @staticmethod
    def top_corrected_associations():
        os.makedirs(GTEx_directory + '/results/{}'.format(group), exist_ok=True)
        from pyensembl import EnsemblRelease
        data = EnsemblRelease(77)

        N = 500
        M = 2000
        k = 1


        all_filt_features, most_varying_feature_idx, expression, _, transcriptIDs, _, _, _ = filter_features_across_all_patchsizes('Lung', 'retrained', 'mean', N, pc_correction=3)
        filt_expression, filt_transcriptIDs = filter_expression(expression, transcriptIDs, M, k)
        filt_features = all_filt_features[256]

        def get_gene_name(transcript):
            transcript_id = transcript.decode('utf-8').split('.')[0]
            return data.gene_name_of_gene_id(transcript_id)



        def get_t_f_idx(position, M):
            f = int(np.floor(position / M))
            t = position % M
            return f, t

        def display_scatter(f, t, axis=None):
            R, pv = pearsonr(filt_features[:,f], filt_expression[:,t])
            if axis:
                axis.scatter(filt_features[:,f], filt_expression[:,t])
                axis.set_title("R: {:0.2} pv: {:0.2}".format(R,pv))
            else:
                plt.scatter(filt_features[:,f], filt_expression[:,t])
            return R, pv


        association_results, assoc_most_varying_feature_idx, assoc_filt_transcriptIDs = pickle.load(open(GTEx_directory + '/intermediate_results/CorrectedFeatureAssociations/corrected_pvalues.pickle', 'rb'))
        Rs_real, pvs_real, pvs_1, pvs_2, pvs_3 = association_results['Lung_mean_retrained_256_3']


        sorted_idx = np.argsort(Rs_real.flatten()**2)[::-1]
        import pdb; pdb.set_trace()







        top10associations = []
        for i in range(10):
            position = sorted_idx[i]
            f, t = get_t_f_idx(position, M)
            print (f,t)
            feature = filt_features[:,f]
            transcript = filt_expression[:,t]
            R, pv = pearsonr(filt_features[:,f], filt_expression[:,t])
            transcript_name = get_gene_name(filt_transcriptIDs[t])
            feature_name = most_varying_feature_idx[f] + 1
            association_data = [feature, feature_name, transcript, transcript_name, pv, R]
            top10associations.append(association_data)

        pickle.dump(top10associations, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))




    @staticmethod
    def raw_associations_across_patchsizes():

        import statsmodels.stats.multitest as smm
        os.makedirs(GTEx_directory + '/results/{}'.format(group), exist_ok=True)


        print ("Loading association data")
        association_results, most_varying_feature_idx, filt_transcriptIDs = pickle.load(open(GTEx_directory + '/intermediate_results/CorrectedFeatureAssociations/corrected_pvalues.pickle', 'rb'))



        SIZES = [128, 256, 512, 1024, 2048, 4096]
        ALPHAS = [0.01,0.0001,0.000001]

        print ("Calculating Bonferroni significant associations:")
        all_counts = []
        for alph in ALPHAS:
            print ("Alpha: ", alph)
            size_counts = []
            for s in SIZES:
                print ("Patch size: ", s)
                pvalues = association_results['{}_{}_{}_{}_3'.format('Lung','mean','retrained',s)][1].flatten()
                counts = sum(smm.multipletests(pvalues, method='bonferroni',alpha=alph)[0])
                size_counts.append(counts)
            all_counts.append(size_counts)

        print ("Saving results")
        pickle.dump(all_counts, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))
        import pdb; pdb.set_trace()

    @staticmethod
    def associations_raw_vs_retrained():

        import statsmodels.stats.multitest as smm
        os.makedirs(GTEx_directory + '/results/{}'.format(group), exist_ok=True)

        print ("Loading association data")
        association_results, most_varying_feature_idx, filt_transcriptIDs = pickle.load(open(GTEx_directory + '/intermediate_results/CorrectedFeatureAssociations/corrected_pvalues.pickle', 'rb'))

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
                pvalues = association_results['{}_{}_{}_{}_3'.format('Lung','mean',m,s)][1].flatten()
                counts = sum(smm.multipletests(pvalues, method='bonferroni',alpha=alpha)[0])
                model_counts.append(counts)
            all_counts.append(model_counts)

        print ("Saving results")
        pickle.dump(all_counts, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))


    @staticmethod
    def associations_mean_vs_median():

        import statsmodels.stats.multitest as smm

        os.makedirs(GTEx_directory + '/results/{}'.format(group), exist_ok=True)
        print ("Loading association data")
        association_results, most_varying_feature_idx, filt_transcriptIDs = pickle.load(open(GTEx_directory + '/intermediate_results/CorrectedFeatureAssociations/corrected_pvalues.pickle', 'rb'))

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
                pvalues = association_results['{}_{}_{}_{}_3'.format('Lung',a,'retrained',s)][1].flatten()
                counts = sum(smm.multipletests(pvalues, method='bonferroni',alpha=alpha)[0])
                aggregation_counts.append(counts)
            all_counts.append(aggregation_counts)

        print ("Saving results")
        pickle.dump(all_counts, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))


    @staticmethod
    def features_with_significant_transcripts():

        import statsmodels.stats.multitest as smm

        os.makedirs(GTEx_directory + '/results/{}'.format(group), exist_ok=True)
        print ("Loading association data")
        association_results, most_varying_feature_idx, filt_transcriptIDs = pickle.load(open(GTEx_directory + '/intermediate_results/CorrectedFeatureAssociations/corrected_pvalues.pickle', 'rb'))

        alpha = 0.0001
        SIZES = [128, 256, 512, 1024, 2048, 4096]


        print ("Calculating Bonferroni significant associations:")
        size_counts = []
        for s in SIZES:
            print ("Patch size: ", s)
            pvalues = association_results['{}_{}_{}_{}_3'.format('Lung','mean','retrained',s)][1]
            original_shape = pvalues.shape
            counts = sum(np.sum(smm.multipletests(pvalues.flatten(),method='bonferroni',alpha=alpha)[0].reshape(original_shape),axis=1) > 0)
            size_counts.append(counts)

        print ("Saving results")
        pickle.dump(size_counts, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))


    @staticmethod
    def transcripts_with_significant_features():

        import statsmodels.stats.multitest as smm

        os.makedirs(GTEx_directory + '/results/{}'.format(group), exist_ok=True)
        print ("Loading association data")
        association_results, most_varying_feature_idx, filt_transcriptIDs = pickle.load(open(GTEx_directory + '/intermediate_results/CorrectedFeatureAssociations/corrected_pvalues.pickle', 'rb'))

        alpha = 0.0001
        SIZES = [128, 256, 512, 1024, 2048, 4096]


        print ("Calculating Bonferroni significant associations:")
        size_counts = []
        for s in SIZES:
            print ("Patch size: ", s)
            pvalues = association_results['{}_{}_{}_{}_3'.format('Lung','mean','retrained',s)][1]
            original_shape = pvalues.shape
            counts = sum(np.sum(smm.multipletests(pvalues.flatten(),method='bonferroni',alpha=alpha)[0].reshape(original_shape),axis=0) > 0)
            size_counts.append(counts)

        print ("Saving results")
        pickle.dump(size_counts, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))



if __name__ == '__main__':
    eval(group + '().' + name + '()')
