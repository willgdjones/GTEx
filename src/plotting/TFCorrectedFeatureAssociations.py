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

class TFCorrectedFeatureAssociations():


    @staticmethod
    def associations_across_patchsizes():
        import seaborn as sns
        sns.set_style("dark")

        SIZES = [128, 256, 512, 1024, 2048, 4096]
        ALPHAS = [0.01, 0.001, 0.0001,0.00001]

        pc_corrected_associations_across_patchsizes = pickle.load(open(GTEx_directory + '/results/TFCorrectedFeatureAssociations/associations_across_patchsizes.pickle', 'rb'))



        plt.figure(figsize=(14,10))
        plt.xticks(range(len(SIZES)), SIZES,size=15)
        plt.xlabel('Patch size', size=30)
        plt.tick_params(axis='both', labelsize=30)
        plt.ylabel('Count', size=30)
        colours = ['blue','red','green','yellow']
        for (k, alph) in enumerate(ALPHAS):
            plt.plot(pc_corrected_associations_across_patchsizes[k], c=colours[k],label=alph)

        plt.legend(prop={'size':30})
        os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)

        plt.show()

    @staticmethod
    def association_statistics():

        import seaborn as sns
        sns.set_style("dark")

        associations_raw_vs_retrained = pickle.load(open(GTEx_directory + '/results/TFCorrectedFeatureAssociations/associations_raw_vs_retrained.pickle', 'rb'))
        associations_mean_vs_median = pickle.load(open(GTEx_directory + '/results/TFCorrectedFeatureAssociations/associations_mean_vs_median.pickle', 'rb'))
        features_with_significant_transcripts = pickle.load(open(GTEx_directory + '/results/TFCorrectedFeatureAssociations/features_with_significant_transcripts.pickle', 'rb'))
        transcripts_with_significant_features = pickle.load(open(GTEx_directory + '/results/TFCorrectedFeatureAssociations/transcripts_with_significant_features.pickle', 'rb'))

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
    def tf_feature_selection():
        expression_ordered_choices = pickle.load(open(GTEx_directory + '/results/{group}/tf_feature_selection_expression.pickle'.format(group=group), 'rb'))
        image_feature_ordered_choices = pickle.load(open(GTEx_directory + '/results/{group}/tf_feature_selection_image_features.pickle'.format(group=group), 'rb'))

        #Extract order of importance of technical factors for expression
        expression_var_explained = [x[0] for x in expression_ordered_choices]
        expression_ordered_tfs = [x[1] for x in expression_ordered_choices]

        #Extract order of importance of technical factors for image features
        image_feature_var_explained = [x[0] for x in image_feature_ordered_choices]
        image_feature_ordered_tfs = [x[1] for x in image_feature_ordered_choices]


        f, a = plt.subplots(1,2,figsize=(30,10))
        a[0].plot(expression_var_explained)
        a[0].set_xticks(list(range(len(expression_ordered_tfs))))
        a[0].set_xticklabels(expression_ordered_tfs, rotation=90)
        a[0].set_ylabel('Extra variance explained', size=15)
        a[0].set_title('Technical factors in order of importance to explain expression variation', size=20)



        a[1].plot(image_feature_var_explained)
        a[1].set_xticks(list(range(len(image_feature_ordered_tfs))))
        a[1].set_xticklabels(image_feature_ordered_tfs, rotation=90)
        a[1].set_ylabel('Cumulative variance explained', size=15)
        a[1].set_title('Technical factors in order of importance to explain image feature variation', size=20)


        os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)

        plt.show()

    







if __name__ == '__main__':
    eval(group + '().' + name + '()')
