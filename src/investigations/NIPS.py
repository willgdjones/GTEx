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
args = vars(parser.parse_args())
group = args['group']
name = args['name']
parameter_key = args['params']


def get_gene_name(transcript):
    transcript_id = transcript.decode('utf-8').split('.')[0]
    return data.gene_name_of_gene_id(transcript_id)

def lookup_enrichment(gene_set):
    clean_gene_set = [x for x in gene_set if x is not None]
    gp = GProfiler("GTEx/wj")
    enrichment_results = gp.gprofile(clean_gene_set)
    return enrichment_results


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


        pool = mp.Pool(processes=2)
        pbar = tqdm(total=len(list(ths)))


        print ("Looking up gene enrichments")
        results = [apply(calculate_tf_enrichment, args=(th,), callback=pbar_update) for th in list(ths)]
        enrichment_results = [p.get() for p in results]

        pickle.dump(enrichment_results, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))




if __name__ == '__main__':
    eval(group + '().' + name + '()')
