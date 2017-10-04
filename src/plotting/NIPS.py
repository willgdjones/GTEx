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



class NIPS():

    @staticmethod
    def compare_variance_explained():
        import seaborn as sns
        sns.set_style("dark")

        SIZES = ['128', '256', '512', '1024', '2048', '4096']

        raw_results = pickle.load(open(GTEx_directory + '/results/NIPSQuestion1/calculate_variance_explained.pickle', 'rb'))
        corrected_results = pickle.load(open(GTEx_directory + '/results/NIPSQuestion2/shared_variability.pickle', 'rb'))

        raw_variance_explained = [raw_results['Lung_mean_retrained_{}'.format(s)] for s in SIZES]
        corrected_variance_explained = [corrected_results['Lung_mean_retrained_{}'.format(s)] for s in SIZES]

        plt.plot(raw_variance_explained, label='Raw')
        plt.plot(corrected_variance_explained, label='Corrected')

        plt.title('Lung mean retrained')
        plt.legend(prop={'size':15})

        # plt.xticklabels(SIZES, size=10)
        plt.xticks(range(len(SIZES)), SIZES)
        plt.xlabel('Patch size')
        plt.ylabel('Variance explained')

        os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)

        plt.show()

    @staticmethod
    def gene_ontology_analysis():
        results = pickle.load(open(GTEx_directory + '/results/TFCorrectedFeatureAssociations/gene_ontology_analysis.pickle', 'rb'))
        ontology_results = results['Lung_mean_retrained_256']
        min_pvs = []
        for res in ontology_results:
            min_pv = 1
            for term in res:

                pv = term[2]
                if pv < min_pv:
                    min_pv = pv
            min_pvs.append(min_pv)

        idx = np.argsort(min_pvs)

        sorted_results = np.array(ontology_results)[idx]

        for i in range(3):
            print(tabulate(sorted_results[i]))

        import pdb; pdb.set_trace()



if __name__ == '__main__':
    eval(group + '().' + name + '()')
