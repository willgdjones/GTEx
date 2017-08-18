import os
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import h5py
import argparse
from sklearn.decomposition import PCA
from openslide import open_slide
from matplotlib.colors import Normalize

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


    @staticmethod
    def expression_PCs_vs_image_feature_PCs():

        image_features, expression, donorIDs, transcriptIDs, technical_factors, technical_headers, technical_idx = extract_final_layer_data('Lung', 'retrained', 'median', '256')

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



class RawFeatureAssociations():

    @staticmethod
    def raw_pvalues():
        os.makedirs(GTEx_directory + '/intermediate_results/{}'.format(group), exist_ok=True)
        SIZES = [128, 256, 512, 1024, 2048, 4096]
        AGGREGATIONS = ['mean', 'median']
        MODELS = ['raw', 'retrained']
        N = 500
        M = 5000
        k = 1

        association_results = {}

        for a in AGGREGATIONS:
            for m in MODELS:
                print ('Filtering features for Lung {} {}'.format(m, a))
                all_filt_features, most_varying_feature_idx, expression, _, transcriptIDs, _, _, _ = filter_features_across_all_patchsizes('Lung', m, a, N)
                filt_expression, filt_transcriptIDs = filter_expression(expression, transcriptIDs, M, k)
                for s in SIZES:
                    filt_features = all_filt_features[s]
                    print('Computing: Lung', a, m, s)
                    res = compute_pearsonR(filt_features, filt_expression)

                    association_results['{}_{}_{}_{}'.format('Lung', a, m, s)] = res

        pickle.dump(association_results, open(GTEx_directory + '/intermediate_results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))

    @staticmethod
    def raw_associations_across_patchsizes():

        import statsmodels.stats.multitest as smm
        os.makedirs(GTEx_directory + '/results/{}'.format(group), exist_ok=True)

        association_results = pickle.load(open(GTEx_directory + '/intermediate_results/RawFeatureAssociations/raw_pvalues.pickle', 'rb'))
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

        association_results = pickle.load(open(GTEx_directory + '/intermediate_results/RawFeatureAssociations/raw_pvalues.pickle', 'rb'))

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
        association_results = pickle.load(open(GTEx_directory + '/intermediate_results/RawFeatureAssociations/raw_pvalues.pickle', 'rb'))

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
        association_results = pickle.load(open(GTEx_directory + '/intermediate_results/RawFeatureAssociations/raw_pvalues.pickle', 'rb'))

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
        association_results = pickle.load(open(GTEx_directory + '/intermediate_results/RawFeatureAssociations/raw_pvalues.pickle', 'rb'))

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




class FeatureExploration():

    @staticmethod
    def extracting_tissue_patches():
        os.makedirs(GTEx_directory + '/results/{}'.format(group), exist_ok=True)
        ID = 'GTEX-13FH7-1726'
        tissue = 'Lung'
        patchsize = 512
        slide, mask, slidemarkings = create_tissue_boundary(ID, tissue, patchsize)

        results = [slide, mask, slidemarkings]

        pickle.dump(results, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))


    @staticmethod
    def feature_variation_across_patchsize():
        os.makedirs(GTEx_directory + '/results/{}'.format(group), exist_ok=True)
        patch_sizes = [128, 256, 512, 1024, 2048, 4096]

        all_image_features = {}
        for ps in patch_sizes:
            image_features, _, _, _, _, _, _ = extract_final_layer_data('Lung', 'retrained', 'median', str(ps))
            all_image_features[ps] = image_features

        results = all_image_features


        pickle.dump(results, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))


    @staticmethod
    def feature_variation_concatenated():

        os.makedirs(GTEx_directory + '/results/{}'.format(group), exist_ok=True)
        patch_sizes = [128, 256, 512, 1024, 2048, 4096]

        all_image_features = {}
        for ps in patch_sizes:
            image_features, _, _, _, _, _, _ = extract_final_layer_data('Lung', 'retrained', 'median', str(ps))
            all_image_features[ps] = image_features

        results = all_image_features

        pickle.dump(results, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))

    @staticmethod
    def expression_means_and_stds():
        _, expression, _, transcriptIDs, _, _, _ = extract_final_layer_data('Lung', 'retrained', 'median', '256')

        results = [expression, transcriptIDs]
        pickle.dump(results, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))



    @staticmethod
    def aggregated_features_across_samples():
        retrained_image_features, _, _, _, _, _, _ = extract_final_layer_data('Lung', 'retrained', 'median', '256')
        raw_image_features, _, _, _, _, _, _ = extract_final_layer_data('Lung', 'raw', 'median', '256')
        results = [retrained_image_features, raw_image_features]
        pickle.dump(results, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))


    @staticmethod
    def features_across_patches():
        os.makedirs(GTEx_directory + '/results/{}'.format(group), exist_ok=True)
        ID1 = 'GTEX-117YW-0526'
        ID2 = 'GTEX-117YX-1326'
        with h5py.File(os.path.join(GTEx_directory, 'data/h5py/collected_features.h5py'), 'r') as f:
            IDlist = list(f['Lung']['-1']['256']['retrained'])
            features1 = f['Lung']['-1']['256']['retrained'][ID1]['features'].value
            features2 = f['Lung']['-1']['256']['retrained'][ID2]['features'].value

        slide1 = open_slide(GTEx_directory + '/data/raw/Lung/{}.svs'.format(ID1))
        image1 = slide1.get_thumbnail(size=(800, 800))

        slide2 = open_slide(GTEx_directory + '/data/raw/Lung/{}.svs'.format(ID2))
        image2 = slide2.get_thumbnail(size=(800, 800))

        results = [features1, image1, features2, image2]

        pickle.dump(results, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))


    @staticmethod
    def feature_crosscorrelation():



        # Filter only non-zero features

        image_features, _, _, _, _, _, _ = extract_final_layer_data('Lung', 'retrained', 'median', '256')

        non_zero_idx = np.std(image_features,axis=0) > 0
        filt_image_features = image_features[:, non_zero_idx]

        def distance(x, y):
            dist = 1 - np.absolute(pearsonr(x,y)[0])
            return dist

        N = filt_image_features.shape[1]

        print ("Calculating features cross correlations")
        D = np.zeros([N,N])
        for i in range(N):
            if i % 100 == 0:
                print (i, '/{}'.format(N))
            for j in range(N):
                dist = distance(filt_image_features[:,i], filt_image_features[:,j])
                if np.isnan(dist) or dist > 1 or dist < 0:
                    import pdb; pdb.set_trace()
                D[i,j] = dist


        pickle.dump(D, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))


class InflationPvalues():


    @staticmethod
    def raw_pvalues():
        os.makedirs(GTEx_directory + '/results/{}'.format(group), exist_ok=True)

        association_results = pickle.load(open(GTEx_directory + '/intermediate_results/RawFeatureAssociations/raw_pvalues.pickle', 'rb'))

        association_results['Lung_']








    # @staticmethod
    # def




    # @staticmethod
    # def feature_crosscorrelation():

    # @staticmethod
    # def features_across_patches():

    # @staticmethod
    # def feature_crosscorrelation():

    # @staticmethod
    # def feature_crosscorrelation():



if __name__ == '__main__':
    eval(group + '().' + name + '()')
