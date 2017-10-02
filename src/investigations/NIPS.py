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
import pyensembl
os.environ['PYENSEMBL_CACHE_DIR'] = '/hps/nobackup/research/stegle/users/willj/GTEx'
import statsmodels.stats.multitest as smm
from pyensembl import EnsemblRelease
from gprofiler import GProfiler
data = EnsemblRelease(77)
import multiprocess as mp
from tqdm import tqdm



GTEx_directory = '/hps/nobackup/research/stegle/users/willj/GTEx'

os.environ['PYENSEMBL_CACHE_DIR'] = GTEx_directory

parser = argparse.ArgumentParser(description='Collection of experiments. Runs on the cluster.')
parser.add_argument('-g', '--group', help='Experiment group', required=True)
parser.add_argument('-n', '--name', help='Experiment name', required=True)
args = vars(parser.parse_args())
group = args['group']
name = args['name']


def get_gene_name(transcript):
    transcript_id = transcript.decode('utf-8').split('.')[0]
    return data.gene_name_of_gene_id(transcript_id)

def lookup_enrichment(gene_set):
    clean_gene_set = [x for x in gene_set if x is not None]
    gp = GProfiler("GTEx/wj")
    enrichment_results = gp.gprofile(clean_gene_set)
    return enrichment_results


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
                        N = pca_image_features.shape[1]
                        M = pca_expression.shape[1]
                        R_matrix = np.zeros(shape=(N,M))
                        for i in range(N):
                            for j in range(M):
                                R, pv = pearsonr(pca_image_features[:,i], pca_expression[:,j])
                                R_matrix[i,j] = R

                        print ("Calculating variance explained")
                        variance_explained = pca_exp.explained_variance_

                        #sum(R_matrix[k,:]) ~ 1 for all k.
                        total = sum([variance_explained[k] * sum(R_matrix[:,k]**2) for k in range(len(variance_explained))])


                        print (total)

                        results[key] = total


        pickle.dump(results, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))




class NIPSQuestion2():
    @staticmethod
    def shared_variability():

        os.makedirs(GTEx_directory + '/results/{}'.format(group), exist_ok=True)

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

                        print (key)

                        Y, X, dIDs, tIDs, tfs, ths, t_idx = extract_final_layer_data(t, m, a, s)
                        Y_prime = Y[t_idx,:]
                        X_prime = X[t_idx,:]
                        tf_idx = [list(ths).index(x) for x in TFs]

                        # Take log of SMTSISH
                        tfs[tfs.index('SMTSISCH')] = np.log2(tfs[tfs.index('SMTSISCH')] + 1)
                        tf_predictors = tfs[:,tf_idx]
                        # tf_predictors = np.array([t[np.random.permutation(tf_predictors_raw.shape[0])] for t in tf_predictors.T]).T

                        lr_X = LinearRegression()


                        # Regress out effects from X
                        lr_X.fit(tf_predictors, X_prime)
                        X_predicted = lr_X.predict(tf_predictors)
                        corrected_X = X_prime - X_predicted


                        lr_Y = LinearRegression()

                        # Regress out effects from X
                        lr_Y.fit(tf_predictors, Y_prime)
                        Y_predicted = lr_Y.predict(tf_predictors)
                        corrected_Y = Y_prime - Y_predicted

                        print ("Calculating PCs that explain 99.9% of corrected image feature variance")
                        pca_Y = PCA(n_components=0.999)
                        pca_corrected_Y = pca_Y.fit_transform(corrected_Y)


                        print ("Calculating PCs that explain 99.9% of expression variance")
                        pca_X = PCA(n_components=0.999)
                        pca_corrected_X = pca_X.fit_transform(corrected_X)

                        print ("Computing correlation matrix")
                        N = pca_corrected_Y.shape[1]
                        M = pca_corrected_X.shape[1]
                        R_matrix = np.zeros(shape=(N,M))
                        for i in range(N):
                            for j in range(M):
                                R, pv = pearsonr(pca_corrected_Y[:,i], pca_corrected_X[:,j])
                                R_matrix[i,j] = R

                        print ("Calculating variance explained")
                        variance_explained = pca_X.explained_variance_


                        total = sum([variance_explained[k] * sum(R_matrix[:,k]**2) for k in range(len(variance_explained))])

                        print (total)

                        results[key] = total


        pickle.dump(results, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))




class NIPSQuestion3A():


    @staticmethod
    def gene_expression_variability():

        os.makedirs(GTEx_directory + '/results/{}'.format(group), exist_ok=True)
        final_results = {}
        
        Y, X, dIDs, tIDs, tfs, ths, t_idx = extract_final_layer_data('Lung', 'retrained', 'mean', '256')
        tfs[list(ths).index('SMTSISCH')] = np.log2(tfs[list(ths).index('SMTSISCH')] + 1)

        tf_X = X[t_idx,:]

        def pbar_update(x):
            pbar.update(1)

        def calculate_tf_enrichment(th):

            i = list(ths).index(th)
            tf = tfs[:, i].reshape(-1,1)


            Rs = []
            pvs = []
            for t in range(tf_X.shape[1]):
                R, pv = pearsonr(tf.flatten(), tf_X[:,t])
                Rs.append(R)
                pvs.append(pv)


            significant_transcripts = tIDs[smm.multipletests(pvs, method='bonferroni',alpha=0.01)[0]]

            significant_genes = []
            for t in significant_transcripts:
                try:
                    g = get_gene_name(t)
                except ValueError:
                    g = None
                significant_genes.append(g)

            enrichments = lookup_enrichment(significant_genes)
            return enrichments


        pool = mp.Pool(processes=16)
        pbar = tqdm(total=len(list(ths)))



        results = [pool.apply_async(calculate_tf_enrichment, args=(th,), callback=pbar_update) for th in list(ths)]
        enrichment_results = [p.get() for p in results]

        final_results[key] = enrichment_results

        pickle.dump(final_results, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))









if __name__ == '__main__':
    eval(group + '().' + name + '()')
