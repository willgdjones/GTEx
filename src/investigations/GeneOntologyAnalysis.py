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
from pebble import ProcessPool, ProcessExpired
from concurrent.futures import TimeoutError

GTEx_directory = '/hps/nobackup/research/stegle/users/willj/GTEx'

parser = argparse.ArgumentParser(description='Collection of experiments. Runs on the cluster.')
parser.add_argument('-g', '--group', help='Experiment group', required=True)
parser.add_argument('-n', '--name', help='Experiment name', required=True)
parser.add_argument('-p', '--params', help='Parameters')
args = vars(parser.parse_args())
group = args['group']
name = args['name']
parameter_key = args['params']



def lookup_enrichment(gene_set):
    clean_gene_set = [x for x in gene_set if x is not None]
    gp = GProfiler("GTEx/wj")
    enrichment_results = gp.gprofile(clean_gene_set)
    return enrichment_results



def get_gene_name(transcript):
    transcript_id = transcript.decode('utf-8').split('.')[0]
    return data.gene_name_of_gene_id(transcript_id)


class GeneOntologyAnalysis():

    @staticmethod
    def gene_ontology_analysis():

        os.makedirs(GTEx_directory + '/results/{}'.format(group), exist_ok=True)
        print ("Loading association data")
        association_results, filt_transcriptIDs = pickle.load(open(GTEx_directory + '/intermediate_results/TFCorrectedFeatureAssociations/compute_pvalues_{key}.pickle'.format(key=parameter_key), 'rb'))


        print ("Calculating significant transcripts")
        significant_indicies = [smm.multipletests(association_results[1][i,:],method='bonferroni',alpha=0.001)[0] for i in range(1024)]
        significant_counts = [sum(x) for x in significant_indicies]
        significant_transcripts = [filt_transcriptIDs[x] for x in significant_indicies]

        print ("Translating into significant genes")
        significant_genes = []
        for (i, feature_transcripts) in enumerate(significant_transcripts):
            genes = []
            for transcript in feature_transcripts:
                try:
                    g = get_gene_name(transcript)
                except ValueError:
                    g = None

                genes.append(g)
            significant_genes.append(genes)

        print ("Looking up gene enrichments for {}".format(parameter_key))

        pbar = tqdm(total=len(significant_genes))


        with ProcessPool(max_workers=16) as pool:

            future = pool.map(lookup_enrichment, significant_genes, timeout=10)
            future_results = future.result()

            enrichment_results = []
            while True:
                try:
                    result = next(future_results)
                    enrichment_results.append(result)
                    pbar.update(1)
                except StopIteration:
                    break
                except TimeoutError as error:
                    enrichment_results.append(None)
                    pbar.update(1)
                    print("function took longer than %d seconds" % error.args[1])
                except ProcessExpired as error:
                    enrichment_results.append(None)
                    pbar.update(1)
                    print("%s. Exit code: %d" % (error, error.exitcode))
                except Exception as error:
                    enrichment_results.append(None)
                    print("function raised %s" % error)
                    print(error)  # Python's traceback of remote process

        del pool

        pickle.dump(enrichment_results, open(GTEx_directory + '/results/{group}/{name}_{key}.pickle'.format(group=group, name=name, key=parameter_key), 'wb'))





if __name__ == '__main__':
    eval(group + '().' + name + '()')
