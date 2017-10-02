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

GTEx_directory = '/hps/nobackup/research/stegle/users/willj/GTEx'

parser = argparse.ArgumentParser(description='Collection of experiments. Runs on the cluster.')
parser.add_argument('-g', '--group', help='Experiment group', required=True)
parser.add_argument('-n', '--name', help='Experiment name', required=True)
args = vars(parser.parse_args())
group = args['group']
name = args['name']




def lookup_enrichment(gene_set):
    clean_gene_set = [x for x in gene_set if x is not None]
    gp = GProfiler("GTEx/wj")
    enrichment_results = gp.gprofile(clean_gene_set)
    return enrichment_results


def get_gene_name(transcript):
    transcript_id = transcript.decode('utf-8').split('.')[0]
    return data.gene_name_of_gene_id(transcript_id)


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

                lr.fit(trial_predictors, pca_Y)
                residuals = pca_Y - lr.predict(trial_predictors)
                var_explained_per_PC = 1 - (np.var(residuals, axis=0) / np.var(pca_Y, axis=0))


                frac_var_explained = np.dot(PCA_Y.explained_variance_, var_explained_per_PC) / sum(PCA_Y.explained_variance_)


                if frac_var_explained > max_frac_var_explained:
                    max_frac_var_explained, max_choice = frac_var_explained, choice



            print ((max_frac_var_explained, max_choice))
            print ("With {}, can explain {} of variance".format(ths[max_choice], max_frac_var_explained))
            ordered_choices.append((max_frac_var_explained, ths[max_choice]))
            idx[max_choice] = 1

        pickle.dump(ordered_choices, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))

    @staticmethod
    def gene_ontology_analysis():

        # import logging

        # logger = mp.log_to_stderr(logging.DEBUG)


        os.makedirs(GTEx_directory + '/results/{}'.format(group), exist_ok=True)
        print ("Loading association data")
        association_results, filt_transcriptIDs = pickle.load(open(GTEx_directory + '/intermediate_results/TFCorrectedFeatureAssociations/compute_pvalues.pickle', 'rb'))

        TISSUES = ['Lung', 'Artery - Tibial', 'Heart - Left Ventricle', 'Breast - Mammary Tissue', 'Brain - Cerebellum', 'Pancreas', 'Testis', 'Liver', 'Ovary', 'Stomach']
        SIZES = ['128', '256', '512', '1024', '2048', '4096']
        AGGREGATIONS = ['mean', 'median']
        MODELS = ['raw', 'retrained']
        TFs = ['SMTSISCH', 'SMNTRNRT', 'SMEXNCRT', 'SMRIN', 'SMATSSCR']

        all_results = {}

        for t in TISSUES:
            for a in AGGREGATIONS:
                for m in MODELS:
                    for s in SIZES:
                        key = '{}_{}_{}_{}'.format(t,a,m,s)

                        print ("Calculating significant transcripts")
                        significant_indicies = [smm.multipletests(association_results['Lung_mean_retrained_256'][1][i,:],method='bonferroni',alpha=0.01)[0] for i in range(1024)]
                        significant_counts = [sum(x) for x in significant_indicies]
                        significant_transcripts = [filt_transcriptIDs[x] for x in significant_indicies]

                        print ("Translating into significant genes")
                        significant_genes = []
                        for (i, feature_transcripts) in enumerate(significant_transcripts):
                            if i % 100 == 0:
                                print ("Gene set ", i)
                            genes = []
                            for t in feature_transcripts:
                                try:
                                    g = get_gene_name(t)
                                except ValueError:
                                    g = None

                                genes.append(g)
                            significant_genes.append(genes)

                        print ("Looking up gene enrichments")

                        # pool = mp.Pool(processes=8)
                        pbar = tqdm(total=len(significant_genes))
                        def pbar_update(x):
                            pbar.update(1)


                        for gene_set in significant_genes:
                            lookup_enrichment(gene_set)
                            pbar.update(1)
                        # enrichment_results = [p.get() for p in results]
                        # logger.info(results)

                        all_results[key] = enrichment_results

        pickle.dump(all_results, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))






























if __name__ == '__main__':
    eval(group + '().' + name + '()')
