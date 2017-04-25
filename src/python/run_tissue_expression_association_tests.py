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

def main():
    
    [X_matrix, y_pheno] = pickle.load(open('data/processed/association_data/expression/{}/{}/{}/X_y_{}'.format(model_name,tile_size,tissue,agg_method),'rb'))

    if tissue == 'Artery - Tibial':
        tissue_filename = 'Artery_Tibial'
    elif tissue == 'Heart - Left Ventricle':
        tissue_filename = 'Heart_Left_Ventricle'
    elif tissue == 'Breast - Mammary Tissue':
        tissue_filename = 'Breast_Mammary_Tissue'
    elif tissue == 'Brain - Cerebellum':
        tissue_filename = 'Brain_Cerebellum'
    else:
        tissue_filename = tissue

    tissue_expression_filepath = '/nfs/research2/stegle/stegle_secure/GTEx/download/49139/PhenoGenotypeFiles/RootStudyConsentSet_phs000424.GTEx.v6.p1.c1.GRU/ExpressionFiles/phe000006.v2.GTEx_RNAseq.expression-data-matrixfmt.c1/parse_data/44_tissues/GTEx_Data_20150112_RNAseq_RNASeQCv1.1.8_gene_rpkm_{}_normalised_without_inverse_gene_expression.txt'.format(tissue_filename)

    with open(tissue_expression_filepath, 'r') as f:
        expression_table = np.array([x.split('\t') for x in f.read().splitlines()])
    transcript_IDs = [x.split('.')[0] for x in expression_table[:,0][1:]]
    data = EnsemblRelease(77)


    # gene_functions = []
    # for (i,ID) in enumerate(transcript_IDs):
        # try:
            # gene_functions.append(data.gene_by_id(ID).biotype)
        # except ValueError:
            # gene_functions.append('NA')
        # if i % 1000 == 0:
            # print (i)
            
    # protein_coding_IDs = np.array(transcript_IDs)[np.array(gene_functions) == 'protein_coding']
    # protein_coding_indexes = np.array([transcript_IDs.index(x) for x in protein_coding_IDs])

    # p_X_matrix = X_matrix[:,protein_coding_indexes]

    filtered_indexes = np.mean(X_matrix, axis=0) > filter_limit

    f_X_matrix = X_matrix[:,filtered_indexes]
    print (f_X_matrix.shape)
    np.random.seed(42)
    upper_limit = math.floor(f_X_matrix.shape[1] / 100) + 1
    p_values = []
    f_y_pheno = y_pheno[:,component].copy()

    shuffled_idx = list(range(len(f_y_pheno)))
    if shuffle !=  '0':
        random.shuffle(shuffled_idx)
        f_y_pheno_shuffled = f_y_pheno[shuffled_idx]
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

    os.makedirs('data/processed/association_results/expression/{}/{}/{}/{}/pvalues'.format(model_name,tile_size,tissue,agg_method), exist_ok=True)
    pickle.dump([shuffled_idx,p_values], open('data/processed/association_results/expression/{}/{}/{}/{}/pvalues/component{}_sh{}_fl{}.py'.format(model_name,tile_size,tissue,agg_method,component,shuffle,filter_limit),'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate assocation p-values for expression matrix X and image representation matrix y')
    # parser.add_argument('-f','--assoc_filepath', help='Path to X-y association data file', required=True)
    parser.add_argument('-f','--shuffle', help='Toggle shuffle. Anything other than 0 is shuffled and given as I use this as the identifier', required=True)
    parser.add_argument('-c','--component', help='Component of y matrix to take', required=True)
    parser.add_argument('-u','--filter_limit', help='Only take gene with mean expression greater than this value', required=True)
    parser.add_argument('-m','--model_name', help='Model uses to generate the representations', required=True)
    parser.add_argument('-t','--tissue', help='Which tissue to perform the association tests on.', required=True)
    parser.add_argument('-a','--agg_method', help='Aggretation method to use, either mean or median', default='mean')
    parser.add_argument('-s','--tile_size', help='The tile of the patches', default='small')
    args = vars(parser.parse_args())
    shuffle = args['shuffle']
    component = int(args['component'])
    filter_limit = int(args['filter_limit'])
    tissue = args['tissue']
    agg_method = args['agg_method']
    model_name = args['model_name']
    tile_size = args['tile_size']
    assert tile_size == 'small' or tile_size == 'medium' or tile_size == 'large'
    if tile_size == 'small':
        tile_level_index = -1
    elif tile_size == 'medium':
        tile_level_index = -2
    elif tile_size == 'large':
        tile_level_index = -3
    else:
        raise Exception 

    assert agg_method == 'median' or agg_method == 'mean', "Aggregation method needs to be either the mean or median"
    if not os.path.isfile('data/processed/association_results/expression/{}/{}/{}/pvalues/component{}_sh{}_fl{}.py'.format(model_name,tissue,agg_method,component,shuffle,filter_limit)):
        main()
    else:
        print ("Results {} {} {} {} {} {} already exists".format(model_name,tissue,agg_method,component,shuffle,filter_limit))
