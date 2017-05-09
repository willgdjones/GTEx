import pdb
import sys
sys.path = ['/nfs/gns/homes/willj/anaconda3/envs/GTEx/lib/python3.5/site-packages'] + sys.path
import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import pearsonr
from lim.genetics import qtl
from lim.genetics.phenotype import NormalPhenotype
import logging
import argparse
import os
from pyensembl import EnsemblRelease
lim_logging = logging.getLogger('lim')
lim_logging.setLevel(logging.CRITICAL)
import h5py
GTEx_directory = '/hps/nobackup/research/stegle/users/willj/GTEx'

parser = argparse.ArgumentParser(description='Generate assocation p-values for expression matrix X and image representation matrix y')
# parser.add_argument('-f','--assoc_filepath', help='Path to X-y association data file', required=True)
parser.add_argument('-l','--shuffle', help='Shuffle index', required=True)
parser.add_argument('-a','--aggregation', help='Aggregation to use out of mean and median', required=True)
parser.add_argument('-f','--feature', help='Image feature to use as the phenotype', required=True)
parser.add_argument('-s','--size', help='Patch size to use', required=True)

args = vars(parser.parse_args())
shuffle = args['shuffle']
aggregation = args['aggregation']
feature = int(args['feature'])
size = args['size']

print (shuffle, aggregation, feature, size)
# size = '128'
# aggregation = 'median'
# feature = 0
# # shuffle = ['real','1','2','3','4','5']
with h5py.File(GTEx_directory + '/data/retrained_inceptionet_aggregations.hdf5') as f:
    image_features = f['lung/{}/{}'.format(size, aggregation)].value
    expression_matrix = f['lung/{}/{}'.format(size, 'expression')].value
    
    single_feature = image_features[:,feature].copy()

    p_values = []
    shuffled_idx = list(range(len(single_feature)))
    if shuffle != 'real':
        random.shuffle(shuffled_idx)
        single_feature = single_feature[shuffled_idx]
        single_feature = NormalPhenotype(single_feature)
    else:
        single_feature = NormalPhenotype(single_feature)
        
    upper_limit = math.floor(expression_matrix.shape[1] / 100) + 1
    print (upper_limit)
    for i in range(upper_limit):
        small_expression_matrix = expression_matrix[:, 100*i:100*(i+1)]
        G = small_expression_matrix.copy()
        lrt = qtl.scan(single_feature, small_expression_matrix, G, progress=False)
        p_values.extend(lrt.pvalues())
        if i % 1 == 0:
            print ('{} transcripts completed'.format(100*(i+1)))
            
    pickle.dump([shuffled_idx,p_values], open(str(GTEx_directory + '/data/retrained_inception_associations/pvalues-{}-{}-{}-{}.py'.format(size,feature,aggregation,shuffle)),'wb'))
        
            
            
            

