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

class Classifier():

    @staticmethod
    def validation_accuracy_across_patchsize():
        import seaborn

        validation_accuracies = np.loadtxt(GTEx_directory + '/results/{group}/{name}.txt'.format(group=group, name=name))
        plt.plot(validation_accuracies)
        plt.title("Top validation accuracy vs patchsize", size=20)
        plt.ylabel('Validation accuracy', size=15)
        plt.xticks([0, 1, 2, 3, 4, 5], ['128', '256', '512', '1024', '2056', '4096'])
        plt.xlabel('Patch size', size=15)

        os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps', format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png', format='png', dpi=100)
        plt.show()

class PCAFeatureAssociations:

    @staticmethod
    def expression_PCs_vs_TFs():


        [R_matrix, pv_matrix, top5_scatter_results] = pickle.load(open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group,name=name), 'rb'))
        norm = MidPointNorm(midpoint=0)


        plt.figure(figsize=(12,10))
        ax1 = plt.subplot2grid((5,5), (0,0),colspan=5, rowspan=2)
        ax2 = plt.subplot2grid((5,5), (2,0),colspan=5, rowspan=2)


        divider1 = make_axes_locatable(ax1)
        divider2 = make_axes_locatable(ax2)
        cax1 = divider1.append_axes('right', size='2%', pad=0.05)
        cax2 = divider2.append_axes('right', size='2%', pad=0.05)

        im1 = ax1.imshow(R_matrix,norm=norm,cmap=PL.get_cmap("coolwarm"))
        ax1.set_title("Technical factors influence Expression PC1. Pearson R values",size=20)
        ax1.set_xlabel('Expression PCs',size=20)
        ax1.set_ylabel('Technical factors',size=20)
        plt.colorbar(im1,cax=cax1, orientation='vertical')


        im2 = ax2.imshow(-np.log10(pv_matrix),cmap=PL.get_cmap("Reds"))
        ax2.set_title("Technical factors influence Expression PC1. -log10 pvalues",size=20)
        ax2.set_xlabel('Expression PCs',size=20)
        ax2.set_ylabel('Technical factors',size=20)
        plt.colorbar(im2, cax=cax2, orientation='vertical')


        import seaborn
        for k in range(5):
            [tf_name, pc_number, tf_vector, pc_vector, R, pv] = top5_scatter_results[k]
            axk = plt.subplot2grid((5,5), (4,k),colspan=1)
            axk.scatter(tf_vector, pc_vector,s=2)
            axk.set_title('R: {:.2}, pv: {:.1}'.format(R, pv))
            axk.set_ylabel('Expression PC {}'.format(pc_number))
            axk.set_xlabel(tf_name)

        plt.tight_layout()


        plt.savefig(GTEx_directory + '/figures/associations/technical_factors_vs_pca_expression.eps',format='eps', dpi=100)

        plt.show()


    @staticmethod
    def image_feature_PCs_vs_TFs():


        [R_matrix, pv_matrix, top5_scatter_results] = pickle.load(open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group,name=name), 'rb'))
        norm = MidPointNorm(midpoint=0)


        fig = plt.figure(figsize=(8,10))
        ax1 = plt.subplot2grid((5,3), (0,0),colspan=1, rowspan=5)
        ax2 = plt.subplot2grid((5,3), (0,1),colspan=1, rowspan=5)


        divider1 = make_axes_locatable(ax1)
        divider2 = make_axes_locatable(ax2)
        cax1 = divider1.append_axes('right', size='2%', pad=0.05)
        cax2 = divider2.append_axes('right', size='2%', pad=0.05)

        # fig.suptitle("Technical factors influence Image feature PC1", y=1.05, size=20)

        # fig.get_axes()[0].annotate('Technical factors influence Image feature PC1', (0.5, 1.05),
        #                     xycoords='figure fraction', ha='center',
        #                     fontsize=24
        #                     )

        im1 = ax1.imshow(R_matrix[:,0:10],norm=norm,cmap=PL.get_cmap("coolwarm"))
        ax1.set_title("Pearson R values",size=10)
        ax1.set_xlabel('Expression PCs',size=20)
        ax1.set_ylabel('Technical factors',size=20)
        fig.colorbar(im1,cax=cax1, orientation='vertical')


        im2 = ax2.imshow(-np.log10(pv_matrix[:,0:10]),cmap=PL.get_cmap("Reds"))
        ax2.set_title("-log10 pvalues",size=10)
        ax2.set_xlabel('Image Feature PCs',size=20)
        ax2.set_ylabel('Technical factors',size=20)
        fig.colorbar(im2, cax=cax2, orientation='vertical')


        import seaborn
        for k in range(5):
            [tf_name, pc_number, tf_vector, pc_vector, R, pv] = top5_scatter_results[k]
            axk = plt.subplot2grid((5,3), (k,2),rowspan=1,colspan=1)
            axk.scatter(tf_vector, pc_vector,s=2)
            axk.set_title('R: {:.2}, pv: {:.1}'.format(R, pv))
            axk.set_ylabel('Image Feature PC {}'.format(pc_number))
            axk.set_xlabel(tf_name)


        plt.tight_layout()
        # plt.subplots_adjust(top=2)

        plt.savefig(GTEx_directory + '/figures/associations/technical_factors_vs_pca_expression.eps',format='eps', dpi=100)
        plt.show()




if __name__ == '__main__':
    eval(group + '().' + name + '()')
