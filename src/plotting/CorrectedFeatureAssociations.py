import os
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import h5py
import argparse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils.helpers import *
import pylab as PL
import matplotlib
from matplotlib import cbook
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize


GTEx_directory = '.'

parser = argparse.ArgumentParser(description='Collection of plotting results. Runs on local computer.')
parser.add_argument('-g', '--group', help='Plotting group', required=True)
parser.add_argument('-n', '--name', help='Plotting name', required=True)
args = vars(parser.parse_args())
group = args['group']
name = args['name']



class CorrectedFeatureAssociations():

    @staticmethod
    def corrected_pvalues():
        raw_results = pickle.load(open(GTEx_directory + '/results/InflationPvalues/raw_pvalues.pickle', 'rb'))
        corrected_results = pickle.load(open(GTEx_directory + '/results/InflationPvalues/corrected_pvalues.pickle', 'rb'))

        import seaborn as sns
        sns.set_style("dark")
        from limix.plot import qqplot

        for (i, pvalues) in enumerate(corrected_results):
            Rs_real, pvs_real, pvs_1, pvs_2, pvs_3 = pvalues
            lamb = estimate_lambda(pvs_real.flatten())
            qqplot(pvs_real.flatten(), label=r'{} PCs, $\lambda={:0.2f}$'.format(i+1, lamb))

        plt.legend(prop={'size':15}, loc='upper left')

        os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)

        plt.show()


    @staticmethod
    def top_corrected_associations():
        top10associations = pickle.load(open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'rb'))

        import seaborn as sns
        sns.set_style("dark")

        fig, ax = plt.subplots(2,5,figsize=(25,8))
        for i in range(10):
            feature, feature_name, transcript, transcript_name, pv, R = top10associations[i]
            ax.flatten()[i].scatter(feature, transcript)
            ax.flatten()[i].set_title("R: {:0.2} pv: {:0.2} {}".format(R, pv, transcript_name), size=15)
            ax.flatten()[i].set_xlabel("Image feature {}".format(feature_name), size=15)
            ax.flatten()[i].set_ylabel("Bulk RNA expression", size=15)

        plt.tight_layout()
        plt.show()


    @staticmethod
    def raw_associations_across_patchsizes():
        import seaborn as sns
        sns.set_style("dark")

        SIZES = [128, 256, 512, 1024, 2048, 4096]
        ALPHAS = [0.01, 0.0001, 0.000001]

        all_counts = pickle.load(open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'rb'))

        plt.figure(figsize=(14,10))
        plt.xticks(range(len(SIZES)), SIZES,size=15)
        plt.xlabel('Patch size', size=30)
        plt.tick_params(axis='both', labelsize=30)
        plt.ylabel('Count', size=30)
        colours = ['blue','red','green']
        for (k, alph) in enumerate(ALPHAS):
            plt.plot(all_counts[k], c=colours[k],label=alph)

        plt.legend(prop={'size':30})
        os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)

        plt.show()

    @staticmethod
    def associations_raw_vs_retrained():

        import seaborn as sns
        sns.set_style("dark")

        all_counts = pickle.load(open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'rb'))

        alpha = 0.0001
        SIZES = [128, 256, 512, 1024, 2048, 4096]
        MODELS = ['retrained', 'raw']

        plt.figure(figsize=(16,10))
        plt.xticks(range(len(SIZES)), SIZES, size=15)
        # plt.xlabel('Patch size', size=60)
        plt.tick_params(axis='both', labelsize=50)
        # plt.ylabel('Count', size=60)
        colours = ['blue','red']
        for (k, m) in enumerate(MODELS):
            plt.plot(all_counts[k], c=colours[k],label=m)

        plt.legend(prop={'size':50})
        os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)

        plt.show()

    @staticmethod
    def associations_mean_vs_median():

        import seaborn as sns
        sns.set_style("dark")

        all_counts = pickle.load(open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'rb'))

        alpha = 0.0001
        SIZES = [128, 256, 512, 1024, 2048, 4096]
        AGGREGATIONS = ['mean', 'median']

        plt.figure(figsize=(16,10))
        plt.xticks(range(len(SIZES)), SIZES, size=15)
        plt.tick_params(axis='both', labelsize=50)

        colours = ['blue','red']
        for (k, m) in enumerate(AGGREGATIONS):
            plt.plot(all_counts[k], c=colours[k],label=m)

        plt.legend(prop={'size':50})
        os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)

        plt.show()

    @staticmethod
    def features_with_significant_transcripts():

        import seaborn as sns
        sns.set_style("dark")

        SIZES = [128, 256, 512, 1024, 2048, 4096]

        size_counts = pickle.load(open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'rb'))
        plt.figure(figsize=(16,10))
        # plt.title("Number of features with significant pvalues (Bonf)", size=20)
        plt.xticks(range(len(SIZES)),SIZES,size=15)
        plt.tick_params(axis='both', labelsize=50)
        plt.plot(size_counts)

        os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)

        plt.show()

    @staticmethod
    def transcripts_with_significant_features():

        import seaborn as sns
        sns.set_style("dark")

        SIZES = [128, 256, 512, 1024, 2048, 4096]

        size_counts = pickle.load(open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'rb'))
        plt.figure(figsize=(16,10))

        plt.xticks(range(len(SIZES)),SIZES,size=15)
        plt.tick_params(axis='both', labelsize=50)
        plt.plot(size_counts)

        os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)

        plt.show()




if __name__ == '__main__':
    eval(group + '().' + name + '()')
