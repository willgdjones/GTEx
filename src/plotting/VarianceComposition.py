import os
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import h5py
import argparse
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pylab as PL
import matplotlib
from matplotlib import cbook
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
sys.path.insert(0, os.getcwd())
from src.utils.helpers import *
from tabulate import tabulate


GTEx_directory = '.'

parser = argparse.ArgumentParser(description='Collection of plotting results. Runs on local computer.')
parser.add_argument('-g', '--group', help='Plotting group', required=True)
parser.add_argument('-n', '--name', help='Plotting name', required=True)
args = vars(parser.parse_args())
group = args['group']
name = args['name']

TISSUES = ['Lung', 'Artery - Tibial', 'Heart - Left Ventricle', 'Breast - Mammary Tissue', 'Brain - Cerebellum', 'Pancreas', 'Testis', 'Liver', 'Ovary', 'Stomach']
TISSUE_TITLES = ['Lung', 'Artery', 'Heart', 'Breast', 'Brain', 'Pancreas', 'Testis', 'Liver', 'Ovary', 'Stomach']
SIZES = ['128', '256', '512', '1024', '2048', '4096']
AGGREGATIONS = ['mean', 'median']
MODELS = ['raw', 'retrained']

class VarianceComposition():

    @staticmethod
    def compare_expression_composition_across_tissues():
        results = pickle.load(open(GTEx_directory + '/results/VarianceComposition/calculate_variance_explained.pickle'.format(group=group), 'rb'))

        fig, ax = plt.subplots(2,5, figsize=(20,5))
        for (i, tissue) in enumerate(TISSUES):
            res = [results['{}_{}_{}_{}'.format(tissue, 'mean', 'retrained', s)] for s in SIZES]

            N = 6
            explained = [r[1][0] for r in res]
            technical = [r[1][1] for r in res]
            unexplained = [r[1][2] for r in res]

            ind = np.arange(N)    # the x locations for the groups
            width = 0.35       # the width of the bars: can also be len(x) sequence

            p1 = ax.flatten()[i].bar(ind, explained, width)
            p2 = ax.flatten()[i].bar(ind, technical, width, bottom=explained)
            p3 = ax.flatten()[i].bar(ind, unexplained, width, bottom=list(np.array(technical) + np.array(explained)))
            ax.flatten()[i].set_title(tissue)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def compare_expression_variance_composition_across_tissues_by_aggregation():
        results = pickle.load(open(GTEx_directory + '/results/VarianceComposition/calculate_variance_explained.pickle'.format(group=group), 'rb'))

        fig, ax = plt.subplots(2,5, figsize=(20,5))
        for (i, tissue) in enumerate(TISSUES):
            res_mean = [results['{}_{}_{}_{}'.format(tissue, 'mean', 'retrained', s)] for s in SIZES]
            res_median = [results['{}_{}_{}_{}'.format(tissue, 'median', 'retrained', s)] for s in SIZES]

            N = 6
            explained_mean = [r[1][0] for r in res_mean]
            technical_mean = [r[1][1] for r in res_mean]
            unexplained_mean = [r[1][2] for r in res_mean]

            explained_median = [r[1][0] for r in res_median]
            technical_median = [r[1][1] for r in res_median]
            unexplained_median = [r[1][2] for r in res_median]

            ind = np.arange(N)    # the x locations for the groups
            width = 0.35       # the width of the bars: can also be len(x) sequence

            p11 = ax.flatten()[i].bar(ind, explained_mean, width)
            p11 = ax.flatten()[i].bar(ind+width, explained_median, width)
            p21 = ax.flatten()[i].bar(ind, technical_mean, width, bottom=explained_mean)
            p22 = ax.flatten()[i].bar(ind+width, technical_median, width, bottom=explained_median)

            p31 = ax.flatten()[i].bar(ind, unexplained_mean, width, bottom=list(np.array(technical_mean) + np.array(explained_mean)))
            p32 = ax.flatten()[i].bar(ind+width, unexplained_median, width, bottom=list(np.array(technical_median) + np.array(explained_median)))
            ax.flatten()[i].set_title(tissue)
            ax.flatten()[i].set_xticks(range(len(SIZES) + 1))
            ax.flatten()[i].set_xticklabels(SIZES)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def compare_expression_variance_composition_across_tissues_by_model():
        results = pickle.load(open(GTEx_directory + '/results/VarianceComposition/calculate_variance_explained.pickle'.format(group=group), 'rb'))


        fig, ax = plt.subplots(2,5, figsize=(12,8))
        for (i, tissue) in enumerate(TISSUES):
            res_raw = [results['{}_{}_{}_{}'.format(tissue, 'mean', 'raw', s)] for s in SIZES]
            res_retrained = [results['{}_{}_{}_{}'.format(tissue, 'mean', 'retrained', s)] for s in SIZES]

            import matplotlib as mpl
            label_size = 20
            mpl.rcParams['xtick.labelsize'] = label_size
            mpl.rcParams['ytick.labelsize'] = label_size

            N = 6
            explained_raw = [r[1][0] for r in res_raw]
            technical_raw = [r[1][1] for r in res_raw]
            unexplained_raw = [r[1][2] for r in res_raw]

            explained_retrained = [r[1][0] for r in res_retrained]
            technical_retrained = [r[1][1] for r in res_retrained]
            unexplained_retrained = [r[1][2] for r in res_retrained]

            ind = np.arange(N)    # the x locations for the groups
            width = 0.35       # the width of the bars: can also be len(x) sequence

            p11 = ax.flatten()[i].bar(ind-(width/2), explained_raw, width, color='green', label='explained')
            p11 = ax.flatten()[i].bar(ind+(width/2), explained_retrained, width, color='green')
            p21 = ax.flatten()[i].bar(ind-(width/2), technical_raw, width, bottom=explained_raw, color='blue', label='technical')
            p22 = ax.flatten()[i].bar(ind+(width/2), technical_retrained, width, bottom=explained_retrained, color='blue')

            p31 = ax.flatten()[i].bar(ind-(width/2), unexplained_raw, width, bottom=list(np.array(technical_raw) + np.array(explained_raw)), color='red', label='unexplained')
            p32 = ax.flatten()[i].bar(ind+(width/2), unexplained_retrained, width, bottom=list(np.array(technical_retrained) + np.array(explained_retrained)), color='red')
            ax.flatten()[i].set_title(TISSUE_TITLES[i], size=20)
            ax.flatten()[i].set_xticks(range(len(SIZES)))
            ax.flatten()[i].set_xticklabels(SIZES, size=15, rotation=90)
        plt.legend()
        plt.tight_layout()

        os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)





        plt.show()








if __name__ == '__main__':
    eval(group + '().' + name + '()')
