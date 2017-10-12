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
# import pyensembl
import statsmodels.stats.multitest as smm
# from pyensembl import EnsemblRelease
# from gprofiler import GProfiler
# data = EnsemblRelease(77)
# import multiprocess as mp
from tqdm import tqdm

# from gevent import Timeout
# from gevent import monkey
# monkey.patch_all(thread=False)
# from pebble import ProcessPool
# from concurrent.futures import TimeoutError



GTEx_directory = '.'

os.environ['PYENSEMBL_CACHE_DIR'] = GTEx_directory

parser = argparse.ArgumentParser(description='Collection of experiments. Runs on the cluster.')
parser.add_argument('-g', '--group', help='Experiment group', required=True)
parser.add_argument('-n', '--name', help='Experiment name', required=True)
parser.add_argument('-p', '--params', help='Parameters')
args = vars(parser.parse_args())
group = args['group']
name = args['name']
parameter_key = args['params']

class GeneOntologyAnalysis():

    @staticmethod
    def gene_ontology_analysis():
        t, a, m, s = parameter_key.split('_')
        enrichment_results = pickle.load(open(GTEx_directory + '/results/{group}/{name}_{key}.pickle'.format(group=group, name=name, key=parameter_key), 'rb'))
        print([len(x) for x in enrichment_results if x is not None])
        import pdb; pdb.set_trace()


















if __name__ == '__main__':
    eval(group + '().' + name + '()')
