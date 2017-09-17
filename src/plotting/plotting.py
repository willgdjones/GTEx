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


GTEx_directory = '.'

parser = argparse.ArgumentParser(description='Collection of plotting results. Runs on local computer.')
parser.add_argument('-g', '--group', help='Plotting group', required=True)
parser.add_argument('-n', '--name', help='Plotting name', required=True)
args = vars(parser.parse_args())
group = args['group']
name = args['name']

class Classifier():

    @staticmethod
    def validation_accuracy_across_patchsize():
        import matplotlib as mpl
        import seaborn as sns
        sns.set_style("dark")

        validation_accuracies = np.loadtxt(GTEx_directory + '/results/{group}/{name}.txt'.format(group=group, name=name))
        fig = plt.figure(figsize=(5,4))
        plt.plot(validation_accuracies)
        plt.ylabel('Validation accuracy', size=15)
        plt.xticks([0, 1, 2, 3, 4, 5], ['128', '256', '512', '1024', '2056', '4096'])
        plt.xlabel('Patch size', size=15)

        label_size = 100
        mpl.rcParams['xtick.labelsize'] = label_size
        mpl.rcParams['ytick.labelsize'] = label_size

        os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)
        plt.show()



class InflationPvalues():

    @staticmethod
    def raw_pvalues():
        import seaborn as sns
        sns.set_style("dark")

        results = pickle.load(open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'rb'))
        Rs_real, pvs_real, pvs_1, pvs_2, pvs_3 = results

        from limix.plot import qqplot

        qqplot(pvs_real.flatten())
        qqplot(pvs_1.flatten())
        qqplot(pvs_2.flatten())
        qqplot(pvs_3.flatten())

        os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)

        plt.show()


    @staticmethod
    def raw_vs_corrected_pvalues():

        print("Loading pvalues")
        raw_results = pickle.load(open(GTEx_directory + '/results/InflationPvalues/raw_pvalues.pickle', 'rb'))
        corrected_results = pickle.load(open(GTEx_directory + '/results/InflationPvalues/corrected_pvalues.pickle', 'rb'))

        import seaborn as sns
        sns.set_style("dark")
        from limix.plot import qqplot

        _, pvs_real_raw, _, _, _ = raw_results
        _, pvs_real_corrected, _, _, _ = corrected_results[0]

        print('Estimating lambda for raw pvalues')
        raw_lamb = estimate_lambda(pvs_real_raw.flatten())
        print('Estimating lambda for corrected pvalues')
        corrected_lamb =  estimate_lambda(pvs_real_corrected.flatten())

        print('Plotting raw pvalues')
        qqplot(pvs_real_raw.flatten(), label='raw $\lambda={:0.2f}$'.format(raw_lamb))
        print('Plotting corrected pvalues')

        qqplot(pvs_real_corrected.flatten(), label='corrected $\lambda={:0.2f}$'.format(corrected_lamb))
        plt.legend(prop={'size':15})

        os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)

        plt.show()


if __name__ == '__main__':
    eval(group + '().' + name + '()')
