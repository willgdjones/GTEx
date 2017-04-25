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
lim_logging = logging.getLogger('lim')
lim_logging.setLevel(logging.CRITICAL)

def main():
    [X_matrix, y_pheno] = pickle.load(open(assoc_filepath,'rb'))

    filtered_indexes = np.mean(X_matrix, axis=0) > filter_limit

    f_X_matrix = X_matrix[:,filtered_indexes]
    print (f_X_matrix.shape)
    np.random.seed(42)
    upper_limit = math.floor(f_X_matrix.shape[1] / 100) + 1
    p_values = []
    f_y_pheno = y_pheno[:,component].copy()

    idx = list(range(len(f_y_pheno)))
    if shuffle == '1':
        random.shuffle(idx)
        f_y_pheno_shuffled = f_y_pheno[idx]
        f_y_pheno_obj = NormalPhenotype(f_y_pheno_shuffled)
    else:
        f_y_pheno_obj = NormalPhenotype(f_y_pheno)

    for i in range(upper_limit):
        sm_X_matrix = f_X_matrix[:, 100*i:100*(i+1)]
        G = sm_X_matrix.copy()
        lrt = qtl.scan(f_y_pheno_obj, sm_X_matrix, G, progress=False)
        p_values.extend(lrt.pvalues())
        if i % 1 == 0:
            print (i)

    os.makedirs('data/processed/association_data/{}/{}/'.format(model_name,tissue), exist_ok=True)
    pickle.dump([idx,p_values], open('data/processed/association_data/{}_pvalues/{}_pvalues_shuffle{}_filterlimit{}.py'.format(filename,component,shuffle,filter_limit),'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate assocation p-values for expression matrix X and image representation matrix y')
    parser.add_argument('-f','--assoc_filepath', help='Path to X-y association data file', required=True)
    parser.add_argument('-s','--shuffle', help='Toggle shuffle', required=True)
    parser.add_argument('-c','--component', help='Component of y matrix to take', required=True)
    parser.add_argument('-u','--filter_limit', help='only take gene with mean expression greater than this value', required=True)
    args = vars(parser.parse_args())
    assoc_filepath = args['assoc_filepath']
    filename = assoc_filepath.split('/')[-1].split('.')[0]
    shuffle = args['shuffle']
    component = int(args['component'])
    filter_limit = int(args['filter_limit'])
    main()
