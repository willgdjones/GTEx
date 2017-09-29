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
    def compute_pvalues():

        os.makedirs(GTEx_directory + '/intermediate_results/{}'.format(group), exist_ok=True)
        SIZES = ['128', '256', '512', '1024', '2048', '4096']
        AGGREGATIONS = ['mean', 'median']
        MODELS = ['raw', 'retrained']

        N = 500
        M = 2000
        k = 1

        print('Computing technical factor corrected associations')

        association_results = {}
        most_varying_feature_indexes = {}

        for a in AGGREGATIONS:
            for m in MODELS:
                for s in SIZES:
                    key = '{}_{}_{}_{}'.format('Lung', a, m, s)
                    Y, X, dIDs, filt_tIDs, tfs, ths, t_idx = filter_and_correct_expression_and_image_features('Lung', m, a, s, M, k, pc_correction=False, tf_correction=True)
                    N = Y.shape[1]
                    print('Computing {} x {} = {} associations for: Lung'.format(N, M, N*M), a, m, s)
                    res = compute_pearsonR(Y, X)
                    association_results[key] = res


        results = [association_results, filt_tIDs]
        pickle.dump(results, open(GTEx_directory + '/intermediate_results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))

    @staticmethod
    def associations_across_patchsizes():

        import statsmodels.stats.multitest as smm
        os.makedirs(GTEx_directory + '/results/{}'.format(group), exist_ok=True)


        print ("Loading association data")
        association_results, most_varying_feature_idx, filt_transcriptIDs = pickle.load(open(GTEx_directory + '/intermediate_results/TFCorrectedFeatureAssociations/corrected_pvalues.pickle', 'rb'))



        SIZES = [128, 256, 512, 1024, 2048, 4096]
        ALPHAS = [0.01, 0.001, 0.0001,0.00001]

        print ("Calculating Bonferroni significant associations:")
        all_counts = []
        for alph in ALPHAS:
            print ("Alpha: ", alph)
            size_counts = []
            for s in SIZES:
                print ("Patch size: ", s)
                pvalues = association_results['{}_{}_{}_{}'.format('Lung','mean','retrained',s)][1].flatten()
                counts = sum(smm.multipletests(pvalues, method='bonferroni',alpha=alph)[0])
                size_counts.append(counts)
            all_counts.append(size_counts)

        print ("Saving results")
        pickle.dump(all_counts, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))

    @staticmethod
    def associations_raw_vs_retrained():

        import statsmodels.stats.multitest as smm
        os.makedirs(GTEx_directory + '/results/{}'.format(group), exist_ok=True)

        print ("Loading association data")
        association_results, most_varying_feature_idx, filt_transcriptIDs = pickle.load(open(GTEx_directory + '/intermediate_results/TFCorrectedFeatureAssociations/corrected_pvalues.pickle', 'rb'))

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
                pvalues = association_results['{}_{}_{}_{}'.format('Lung','mean',m,s)][1].flatten()
                counts = sum(smm.multipletests(pvalues, method='bonferroni',alpha=alpha)[0])
                model_counts.append(counts)
            all_counts.append(model_counts)

        print ("Saving results")
        pickle.dump(all_counts, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))

    @staticmethod
    def top_corrected_associations():
        os.makedirs(GTEx_directory + '/results/{}'.format(group), exist_ok=True)
        from pyensembl import EnsemblRelease
        data = EnsemblRelease(77)

        N = 500
        M = 2000
        k = 1


        all_filt_features, most_varying_feature_idx, expression, _, transcriptIDs, tfs, ths, t_idx = filter_features_across_all_patchsizes('Lung', 'retrained', 'mean', N, tf_correction=True)
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


        association_results, assoc_most_varying_feature_idx, assoc_filt_transcriptIDs = pickle.load(open(GTEx_directory + '/intermediate_results/TFCorrectedFeatureAssociations/corrected_pvalues.pickle', 'rb'))
        Rs_real, pvs_real, pvs_1, pvs_2, pvs_3 = association_results['Lung_mean_retrained_256']


        sorted_idx = np.argsort(Rs_real.flatten()**2)[::-1]


        top10associations = []
        for i in range(10):
            position = sorted_idx[i]
            f, t = get_t_f_idx(position, M)
            print (f,t)
            feature = filt_features[:,f]
            transcript = filt_expression[:,t]

            R, pv = pearsonr(feature, transcript[t_idx])
            transcript_name = get_gene_name(filt_transcriptIDs[t])
            feature_name = most_varying_feature_idx[f] + 1
            association_data = [feature, feature_name, transcript[t_idx], transcript_name, pv, R]
            top10associations.append(association_data)

        pickle.dump(top10associations, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))



    @staticmethod
    def associations_mean_vs_median():

        import statsmodels.stats.multitest as smm

        os.makedirs(GTEx_directory + '/results/{}'.format(group), exist_ok=True)
        print ("Loading association data")
        association_results, most_varying_feature_idx, filt_transcriptIDs = pickle.load(open(GTEx_directory + '/intermediate_results/TFCorrectedFeatureAssociations/corrected_pvalues.pickle', 'rb'))

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
        print ("Loading association data")
        association_results, most_varying_feature_idx, filt_transcriptIDs = pickle.load(open(GTEx_directory + '/intermediate_results/TFCorrectedFeatureAssociations/corrected_pvalues.pickle', 'rb'))

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
        print ("Loading association data")
        association_results, most_varying_feature_idx, filt_transcriptIDs = pickle.load(open(GTEx_directory + '/intermediate_results/TFCorrectedFeatureAssociations/corrected_pvalues.pickle', 'rb'))

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
    def tf_feature_selection():
        Y, X, dIDs, tIDs, tfs, ths, t_idx = extract_final_layer_data('Lung', 'retrained', 'mean', '256')

        TFs = ['SMTSISCH', 'SMNTRNRT', 'SMEXNCRT', 'SMRIN', 'SMATSSCR']
        tf_Y = Y[t_idx,:]
        tf_X = X[t_idx,:]

        tfs[:,TFs.index('SMTSISCH')] = np.log2(tfs[:,TFs.index('SMTSISCH')] + 1)
        tf_idx = [list(ths).index(x) for x in TFs]
        tf_predictors = tfs[:,tf_idx]

        print ("Calculating PCs that explain 99.9% of image feature variance")
        PCA_Y = PCA(n_components=0.999)
        pca_Y = PCA_Y.fit_transform(tf_Y)


        print ("Calculating PCs that explain 99.9% of expression variance")
        PCA_X = PCA(n_components=0.999)
        pca_X = PCA_X.fit_transform(tf_X)

        idx = np.zeros_like(ths)
        N = pca_X.shape[1]
        ordered_choices = []

        print ("Performing feature selection")

        for i in range(51):
            print ("Iteration {}".format(i))

            selected_c = np.argwhere(idx == 1).flatten()
            tf_predictors = tfs[:,selected_c]
            lr = LinearRegression()
            pca_X_copy = pca_X.copy()
            if i > 0:
                lr.fit(tf_predictors, pca_X)
                regressed_pca_X = lr.predict(tf_predictors)
                regressed_pca_X = pca_X_copy - regressed_pca_X
            else:
                regressed_pca_X = pca_X_copy

            maxvarexplained, max_c = [0, None]
            unselected_c = np.argwhere(idx == 0).flatten()
            print ("{} choices left".format(len(unselected_c)))
            for c in unselected_c:
                varexplained = np.dot(PCA_X.explained_variance_, [pearsonr(regressed_pca_X[:,j], tfs[:, c])[0]**2 for j in range(N)])

                if varexplained > maxvarexplained:
                    maxvarexplained, max_c = varexplained, c



            print ((maxvarexplained, max_c))
            print ("Choosing {}, explains {} variance".format(ths[max_c], maxvarexplained))
            ordered_choices.append((maxvarexplained, ths[max_c]))
            idx[max_c] = 1

        pickle.dump(ordered_choices, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))

        import pdb; pdb.set_trace()














if __name__ == '__main__':
    eval(group + '().' + name + '()')
