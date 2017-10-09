import os
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import h5py
import argparse
from mpl_toolkits.axes_grid1 import make_axes_locatable
sys.path.insert(0, os.getcwd())
from src.utils.helpers import *
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




class RawFeatureAssociations():

    @staticmethod
    def associations_across_patchsizes():
        import seaborn as sns
        sns.set_style("dark")

        SIZES = [128, 256, 512, 1024, 2048, 4096]
        ALPHAS = [0.01, 0.0001, 0.000001]

        raw_associations_across_patchsizes = pickle.load(open(GTEx_directory + '/results/RawFeatureAssociations/raw_associations_across_patchsizes.pickle', 'rb'))



        plt.figure(figsize=(14,10))
        plt.xticks(range(len(SIZES)), SIZES,size=15)
        plt.xlabel('Patch size', size=30)
        plt.tick_params(axis='both', labelsize=30)
        plt.ylabel('Count', size=30)
        colours = ['blue','red','green']
        for (k, alph) in enumerate(ALPHAS):
            plt.plot(raw_associations_across_patchsizes[k], c=colours[k],label=alph)

        plt.legend(prop={'size':30})
        os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)

        plt.show()

    @staticmethod
    def association_statistics():

        import seaborn as sns
        sns.set_style("dark")

        associations_raw_vs_retrained = pickle.load(open(GTEx_directory + '/results/RawFeatureAssociations/associations_raw_vs_retrained.pickle', 'rb'))
        associations_mean_vs_median = pickle.load(open(GTEx_directory + '/results/RawFeatureAssociations/associations_mean_vs_median.pickle', 'rb'))
        features_with_significant_transcripts = pickle.load(open(GTEx_directory + '/results/RawFeatureAssociations/features_with_significant_transcripts.pickle', 'rb'))
        transcripts_with_significant_features = pickle.load(open(GTEx_directory + '/results/RawFeatureAssociations/transcripts_with_significant_features.pickle', 'rb'))

        ALPHA = 0.0001
        SIZES = [128, 256, 512, 1024, 2048, 4096]
        MODELS = ['retrained', 'raw']
        AGGREGATIONS = ['mean', 'median']

        fig, axes = plt.subplots(1,4,figsize=(20,3))

        # associations_raw_vs_retrained

        axes[0].set_xticklabels(SIZES, size=10)
        axes[0].set_xticks(range(len(SIZES)))
        axes[0].set_ylabel("Number of associations")

        # axes[0].tick_params(axis='both')

        colours = ['blue','red']
        for (k, m) in enumerate(MODELS):
            axes[0].plot(associations_raw_vs_retrained[k], c=colours[k],label=m)

        axes[0].legend()

        # associations_mean_vs_median

        axes[1].set_xticklabels(SIZES, size=10)
        axes[1].set_xticks(range(len(SIZES)))
        axes[1].tick_params(axis='both')

        colours = ['blue','red']
        for (k, m) in enumerate(AGGREGATIONS):
            axes[1].plot(associations_mean_vs_median[k], c=colours[k],label=m)

        axes[1].legend()
        # prop={'size':50}


        # features_with_significant_transcripts
        # plt.title("Number of features with significant pvalues (Bonf)", size=20)
        axes[2].set_xticklabels(SIZES, size=10)
        axes[2].set_xticks(range(len(SIZES)))
        axes[2].plot(features_with_significant_transcripts)



        # transcripts_with_significant_features
        axes[3].set_xticklabels(SIZES, size=10)
        axes[3].set_xticks(range(len(SIZES)))
        axes[3].plot(transcripts_with_significant_features)

        os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)

        plt.show()


    @staticmethod
    def image_feature_796_vs_SMTSISCH():
        import seaborn as sns
        sns.set_style("dark")

        plt.figure(figsize=(10,7))
        results = pickle.load(open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'rb'))
        SMTSISCH, feature796  = results
        plt.xlabel('SMTSISCH',size=40)
        plt.ylabel('Image feature 796',size=40)
        plt.scatter(SMTSISCH, feature796, s=40)
        R, pv = pearsonr(SMTSISCH, feature796)
        plt.title('R: {:.2}, pv: {:.1}'.format(R, pv), size=40)

        os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)

        plt.show()

    @staticmethod
    def image_feature_671_vs_SMTSISCH():
        import seaborn as sns
        sns.set_style("dark")

        plt.figure(figsize=(10,7))
        results = pickle.load(open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'rb'))
        SMTSISCH, feature671  = results
        plt.xlabel('SMTSISCH',size=40)
        plt.ylabel('Image feature 671',size=40)
        plt.scatter(SMTSISCH, feature671, s=40)
        R, pv = pearsonr(SMTSISCH, feature671)
        plt.title('R: {:.2}, pv: {:.1}'.format(R, pv), size=40)

        os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)

        plt.show()



    @staticmethod
    def image_feature_671_vs_TFs():
        import seaborn as sns
        sns.set_style("dark")

        fig, ax, = plt.subplots(1,5, figsize=(25,4))
        results = pickle.load(open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'rb'))

        SMATSSCR, SMNTRNRT, SMTSISCH, SMEXNCRT, feature671, SMRIN = results
        for (i, tf) in enumerate(['SMATSSCR', 'SMNTRNRT', 'SMTSISCH', 'SMEXNCRT', 'SMRIN']):
            R, pv = pearsonr(eval(tf), feature671)
            ax[i].set_title('R: {:.2}, pv: {:.1}'.format(R, pv), size=15)
            ax[i].scatter(eval(tf), feature671)
            ax[i].set_xlabel(tf)
            ax[i].set_ylabel('Image feature 671')

        plt.tight_layout()
        os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)

        plt.show()

    @staticmethod
    def image_feature_796_vs_TFs():
        import seaborn as sns
        sns.set_style("dark")

        fig, ax, = plt.subplots(1,5, figsize=(25,4))
        results = pickle.load(open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'rb'))

        SMATSSCR, SMNTRNRT, SMTSISCH, SMEXNCRT, feature671, SMRIN = results
        for (i, tf) in enumerate(['SMATSSCR', 'SMNTRNRT', 'SMTSISCH', 'SMEXNCRT', 'SMRIN']):
            R, pv = pearsonr(eval(tf), feature671)
            ax[i].set_title('R: {:.2}, pv: {:.1}'.format(R, pv), size=15)
            ax[i].scatter(eval(tf), feature671)
            ax[i].set_xlabel(tf)
            ax[i].set_ylabel('Image feature 796')

        plt.tight_layout()
        os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)

        plt.show()

        # SMTSISCH, feature671  = results
        # plt.xlabel('SMTSISCH',size=40)
        # plt.ylabel('Image feature 671',size=40)
        # plt.scatter(SMTSISCH, feature671, s=40)
        # R, pv = pearsonr(SMTSISCH, feature671)
        # plt.title('R: {:.2}, pv: {:.1}'.format(R, pv), size=40)
        #
        # os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)
        # plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        # plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)
        #
        # plt.show()


if __name__ == '__main__':
    eval(group + '().' + name + '()')
