
import sys
sys.path = ['/nfs/gns/homes/willj/anaconda3/envs/GTEx/lib/python3.5/site-packages'] + sys.path
import pickle
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
from scipy.stats import pearsonr
import os
import argparse


def main():
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

    for component in range(1024):
        print (component)
        graph_filepath = 'data/processed/association_results/expression/{}/{}/{}/mean/graphs/component{}_sh1_fl0.png'.format(model_name,tile_size,tissue,component)
        if os.path.isfile(graph_filepath):
            continue

        real_pvalues = np.array(pickle.load(open('data/processed/association_results/expression/{}/{}/{}/mean/pvalues/component{}_sh0_fl0.py'.format(model_name,tile_size,tissue,component),'rb'))[1])
        shuffled_data = pickle.load(open('data/processed/association_results/expression/{}/{}/{}/mean/pvalues/component{}_sh1_fl0.py'.format(model_name,tile_size,tissue,component),'rb'))
        shuffled_idx = np.array(shuffled_data[0])
        shuffled_pvalues = np.array(shuffled_data[1])
        [X_matrix, y_pheno] = pickle.load(open('data/processed/association_data/expression/{}/{}/{}/X_y_mean'.format(model_name,tile_size,tissue),'rb'))




        sorted_pvalues_idx = np.argsort(real_pvalues)
        print (sorted_pvalues_idx[0:5])


        f,a = plt.subplots(1,5, figsize=(20,4))
        f.suptitle("{}, size: {}, component: {}, idx top 5 pvalues {}".format(tissue, tile_size, component, sorted_pvalues_idx[0:5]))
        titles = []
        for i in range(3):
            a[i].scatter(X_matrix[:,sorted_pvalues_idx[i]], y_pheno[:,component], s=3)
            a[i].set_title("{}".format(expression_table[1:,0][sorted_pvalues_idx[i]]))

        sorted_real_indexes = np.argsort(real_pvalues)
        sorted_real_pvalues = real_pvalues[sorted_real_indexes]
        sorted_shuffled_indexes = np.argsort(shuffled_pvalues)
        sorted_shuffled_pvalues = shuffled_pvalues[sorted_shuffled_indexes]


        sample_real = sorted_real_pvalues
        expected_real = np.linspace(1/len(sorted_real_pvalues), 1, len(sorted_real_pvalues))
        sample_shuffle = sorted_shuffled_pvalues
        expected_shuffle = np.linspace(1/len(sorted_shuffled_pvalues), 1, len(sorted_shuffled_pvalues))

        a[3].scatter([-math.log(x,10) for x in expected_real], [-math.log(x,10) for x in sample_real], s=3)
        a[3].plot(np.linspace(0,8,100), np.linspace(0,8,100),c='red')
        a[3].set_title('Real')
        a[4].scatter([-math.log(x, 10) for x in expected_shuffle], [-math.log(x,10) for x in sample_shuffle], s=3)
        a[4].plot(np.linspace(0,8,100), np.linspace(0,8,100),c='red')
        a[4].set_title('Shuffled')

        os.makedirs('data/processed/association_results/expression/{}/{}/{}/mean/graphs'.format(model_name,tile_size,tissue),exist_ok=True)

        plt.savefig(graph_filepath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-m','--model_name', help='Name of the model', required=True)
    parser.add_argument('-t','--tissue', help='Tissue type', required=True)
    parser.add_argument('-s','--tile_size', help='The tile of the patches', default='small')
    args = vars(parser.parse_args())
    model_name = args['model_name']
    tissue = args['tissue']
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
    main()
