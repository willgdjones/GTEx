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

# from gevent import Timeout
# from gevent import monkey
# monkey.patch_all(thread=False)
from pebble import ProcessPool
from concurrent.futures import TimeoutError



GTEx_directory = '/hps/nobackup/research/stegle/users/willj/GTEx'

os.environ['PYENSEMBL_CACHE_DIR'] = GTEx_directory

parser = argparse.ArgumentParser(description='Collection of experiments. Runs on the cluster.')
parser.add_argument('-g', '--group', help='Experiment group', required=True)
parser.add_argument('-n', '--name', help='Experiment name', required=True)
parser.add_argument('-p', '--params', help='Parameters')
parser.add_argument('-l', '--parallel', help='Parameters')
parser.add_argument('-s', '--shuffle', help='Parameters')
args = vars(parser.parse_args())
group = args['group']
name = args['name']
parameter_key = args['params']
parallel = args['parallel']
shuffle = args['shuffle']

def variance_excluding_technical_factor(key, shuffle=False):
    t, a, m, s = key.split('_')

    Y, X, dIDs, tIDs, tfs, ths, t_idx = extract_final_layer_data(t, m, a, s, shuffle=shuffle)
    Y_prime = Y[t_idx,:]
    X_prime = X[t_idx,:]

    # Take log of SMTSISH
    tfs[list(ths).index('SMTSISCH')] = np.log2(tfs[list(ths).index('SMTSISCH')] + 1)
    tf_predictors = tfs

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

    # print ("Calculating PCs that explain 99.9% of corrected image feature variance")
    pca_Y = PCA(n_components=0.999)
    pca_corrected_Y = pca_Y.fit_transform(corrected_Y)


    # print ("Calculating PCs that explain 99.9% of expression variance")
    pca_X = PCA(n_components=0.999)
    pca_corrected_X = pca_X.fit_transform(corrected_X)

    # print ("Computing correlation matrix")
    N = pca_corrected_Y.shape[1]
    M = pca_corrected_X.shape[1]
    R_matrix = np.zeros(shape=(N,M))
    # pbar = tqdm(total=N*M)
    for i in range(N):
        for j in range(M):
            R, pv = pearsonr(pca_corrected_Y[:,i], pca_corrected_X[:,j])
            R_matrix[i,j] = R
            # pbar.update(1)


    Y_variance_explained = pca_Y.explained_variance_
    X_variance_explained = pca_X.explained_variance_



    # print ("Calculating variance of image features explained by expression ")
    componentwise_variance_explained = []
    for i in range(len(Y_variance_explained)):
        component_variance_explained = Y_variance_explained[i] * sum(R_matrix[i,:]**2)
        componentwise_variance_explained.append(component_variance_explained)



    total = sum(componentwise_variance_explained)
    frac = total / sum(Y_variance_explained)

    image_feature_variation_explained = [total, frac]

    # print ("Calculating variance of expression explained by image features")
    componentwise_variance_explained = []

    for k in range(len(X_variance_explained)):
        component_variance_explained = X_variance_explained[k] * sum(R_matrix[:,k]**2)
        componentwise_variance_explained.append(component_variance_explained)


    total = sum(componentwise_variance_explained)
    frac = total / sum(X_variance_explained)

    expression_variation_explained = [total, frac]


    return image_feature_variation_explained, expression_variation_explained

def variance_including_technical_factor(key, shuffle=False):
    t, a, m, s = key.split('_')

    image_features, expression, donorIDs, transcriptIDs, technical_factors, technical_headers, technical_idx = extract_final_layer_data(t, m, a, s, shuffle=shuffle)

    # print ("Calculating PCs that explain 99.9% of image feature variance")
    pca_im = PCA(n_components=0.999)
    pca_image_features = pca_im.fit_transform(image_features)


    # print ("Calculating PCs that explain 99.9% of expression variance")
    pca_exp = PCA(n_components=0.999)
    pca_expression = pca_exp.fit_transform(expression)

    # print ("Computing correlation matrix")
    N = pca_image_features.shape[1]
    M = pca_expression.shape[1]
    R_matrix = np.zeros(shape=(N,M))
    # pbar = tqdm(total=N*M)
    for i in range(N):
        for j in range(M):
            R, pv = pearsonr(pca_image_features[:,i], pca_expression[:,j])
            R_matrix[i,j] = R
            # pbar.update(1)
    # pbar.close()


    im_variance_explained = pca_im.explained_variance_
    exp_variance_explained = pca_exp.explained_variance_

    # print ("Calculating variance of image features explained by expression ")
    #sum(R_matrix[k,:]) ~ 1 for all k.
    componentwise_variance_explained = []
    for i in range(len(im_variance_explained)):
        component_variance_explained = im_variance_explained[i] * sum(R_matrix[i,:]**2)
        componentwise_variance_explained.append(component_variance_explained)

    total = sum(componentwise_variance_explained)
    frac = total / sum(im_variance_explained)

    image_feature_variation_explained = [total, frac]

    # print ("Calculating variance of expression explained by image features")

    componentwise_variance_explained = []
    for k in range(len(exp_variance_explained)):
        component_variance_explained = exp_variance_explained[k] * sum(R_matrix[:,k]**2)
        componentwise_variance_explained.append(component_variance_explained)

    total = sum(componentwise_variance_explained)
    frac = total / sum(exp_variance_explained)

    expression_variation_explained = [total, frac]

    return image_feature_variation_explained, expression_variation_explained

def compute_variance_composition(key, shuffle=False):
    var_exc_tf = variance_excluding_technical_factor(key, shuffle)
    var_inc_tf = variance_including_technical_factor(key, shuffle)

    total_exp_inc_variation = var_inc_tf[1][0] / var_inc_tf[1][1]
    total_im_inc_variation = var_inc_tf[0][0] / var_inc_tf[0][1]

    total_exp_exc_variation = var_exc_tf[1][0] / var_exc_tf[1][1]
    total_im_exc_variation = var_exc_tf[0][0] / var_exc_tf[0][1]

    im_explained_by_tf = total_im_inc_variation - total_im_exc_variation
    exp_explained_by_tf = total_exp_inc_variation - total_exp_exc_variation

    im_variance_explained_by_exp, im_tech_variance, im_unexplained = var_exc_tf[0][0], im_explained_by_tf, total_im_inc_variation - var_exc_tf[0][0] - im_explained_by_tf
    exp_variance_explained_by_im, exp_tech_variance, exp_unexplained = var_exc_tf[1][0], exp_explained_by_tf, total_exp_inc_variation - var_exc_tf[1][0] - exp_explained_by_tf
    im_components = im_variance_explained_by_exp, im_tech_variance, im_unexplained
    exp_components = exp_variance_explained_by_im, exp_tech_variance, exp_unexplained

    return im_components, exp_components


def compute_variance_composition_real_and_shuffle(key):
    components = compute_variance_composition(key, shuffle=False)
    shuffled_components = compute_variance_composition(key, shuffle=True)
    return components, shuffled_components

class VarianceComposition():

    @staticmethod
    def calculate_variance_explained():

        os.makedirs(GTEx_directory + '/results/{}'.format(group), exist_ok=True)

        TISSUES = ['Lung', 'Artery - Tibial', 'Heart - Left Ventricle', 'Breast - Mammary Tissue', 'Brain - Cerebellum', 'Pancreas', 'Testis', 'Liver', 'Ovary', 'Stomach']
        SIZES = ['128', '256', '512', '1024', '2048', '4096']
        AGGREGATIONS = ['mean']
        MODELS = ['retrained']

        results = {}

        all_keys = []
        for t in TISSUES:
            for a in AGGREGATIONS:
                for m in MODELS:
                    for s in SIZES:
                        key = '{}_{}_{}_{}'.format(t,a,m,s)
                        all_keys.append(key)

        pbar = tqdm(total=len(all_keys))
        results = {}

        if parallel == '1':
            import pdb; pdb.set_trace()
            with ProcessPool(max_workers=16) as pool:

                future = pool.map(compute_variance_composition_real_and_shuffle, all_keys, timeout=600)
                future_results = future.result()

                variance_results = []
                while True:
                    try:
                        result = next(future_results)
                        variance_results.append(result)
                        pbar.update(1)
                    except StopIteration:
                        break
                    except TimeoutError as error:
                        variance_results.append(None)
                        pbar.update(1)
                        print("function took longer than %d seconds" % error.args[1])
                    except ProcessExpired as error:
                        variance_results.append(None)
                        pbar.update(1)
                        print("%s. Exit code: %d" % (error, error.exitcode))
                    except Exception as error:
                        variance_results.append(None)
                        print("function raised %s" % error)
                        print(error)  # Python's traceback of remote process

                results = dict(zip(all_keys, variance_results))
        else:
            for key in all_keys:
                components = compute_variance_composition(key)
                shuffled_components = compute_variance_composition(key, shuffle=True)

                results[key] = components
                pbar.update(1)




        pickle.dump(results, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))


if __name__ == '__main__':
    eval(group + '().' + name + '()')
