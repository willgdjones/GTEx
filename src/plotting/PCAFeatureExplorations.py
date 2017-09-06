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

class PCAFeatureAssociations:

    @staticmethod
    def expression_PCs_vs_TFs():

        [R_matrix, pv_matrix, top5_scatter_results] = pickle.load(open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group,name=name), 'rb'))
        norm = MidPointNorm(midpoint=0)

        def get_axis_limits(ax, scale=.9):
            return ax.get_xlim()[1]*scale, ax.get_ylim()[1]*scale


        fig = plt.figure(figsize=(8,10))
        ax1 = plt.subplot2grid((5,3), (0,0),colspan=1, rowspan=5)
        ax2 = plt.subplot2grid((5,3), (0,1),colspan=1, rowspan=5)


        divider1 = make_axes_locatable(ax1)
        divider2 = make_axes_locatable(ax2)
        cax1 = divider1.append_axes('right', size='5%', pad=0.05)
        cax2 = divider2.append_axes('right', size='5%', pad=0.05)

        im1 = ax1.imshow(R_matrix[:,0:10], norm=norm, cmap=PL.get_cmap("coolwarm"))
        ax1.set_title("Pearson R values", size=15)
        ax1.set_xlabel('Expression PCs', size=15)
        ax1.set_ylabel('Technical factors', size=15)

        ax1.annotate('A', xycoords='axes fraction', xy=(0.8,0.95),size=20, color='green')
        plt.colorbar(im1, cax=cax1, orientation='vertical')



        im2 = ax2.imshow(-np.log10(pv_matrix[:,0:10]),cmap=PL.get_cmap("Reds"))

        ax2.set_title("-log10 pvalues", size=15)
        ax2.set_xlabel('Expression PCs', size=15)
        ax2.set_ylabel('Technical factors', size=15)
        ax2.annotate('B', xycoords='axes fraction', xy=(0.8,0.95),size=20, color='green')
        plt.colorbar(im2, cax=cax2, orientation='vertical')

        import seaborn as sns
        sns.set_style("dark")
        import matplotlib as mpl
        label_size = 7
        mpl.rcParams['xtick.labelsize'] = label_size
        mpl.rcParams['ytick.labelsize'] = label_size


        for (k,l) in enumerate(['C', 'D', 'E', 'F', 'G']):
            [tf_name, pc_number, tf_vector, pc_vector, R, pv] = top5_scatter_results[k]
            axk = plt.subplot2grid((5,3), (k,2),rowspan=1,colspan=1)
            axk.scatter(tf_vector, pc_vector,s=2)
            axk.set_title('R: {:.2}, pv: {:.1}'.format(R, pv), size=10)
            axk.set_ylabel('PC {}'.format(pc_number))
            axk.set_xlabel(tf_name)

            axk.annotate(l, xycoords='axes fraction', xy=(0.8,0.7),size=20, color='green')

        plt.tight_layout()


        # plt.savefig(GTEx_directory + '/figures/associations/technical_factors_vs_pca_expression.eps',format='eps', dpi=100)
        os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)

        plt.show()


    @staticmethod
    def image_feature_PCs_vs_TFs():
        [R_matrix, pv_matrix, top5_scatter_results] = pickle.load(open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group,name=name), 'rb'))
        norm = MidPointNorm(midpoint=0)


        fig = plt.figure(figsize=(8,10))
        ax1 = plt.subplot2grid((5,3), (0,0),colspan=1, rowspan=5)
        ax2 = plt.subplot2grid((5,3), (0,1),colspan=1, rowspan=5)
        ax1.annotate('A', xycoords='axes fraction', xy=(0.8,0.95),size=20, color='green')


        divider1 = make_axes_locatable(ax1)
        divider2 = make_axes_locatable(ax2)
        cax1 = divider1.append_axes('right', size='5%', pad=0.05)
        cax2 = divider2.append_axes('right', size='5%', pad=0.05)


        im1 = ax1.imshow(R_matrix[:,0:10],norm=norm,cmap=PL.get_cmap("coolwarm"))
        ax1.set_title("Pearson R values",size=15)
        ax1.set_xlabel('Image Feature PCs',size=15)
        ax1.set_ylabel('Technical factors',size=15)
        fig.colorbar(im1,cax=cax1, orientation='vertical')


        im2 = ax2.imshow(-np.log10(pv_matrix[:,0:10]),cmap=PL.get_cmap("Reds"))
        ax2.set_title("-log10 pvalues",size=15)
        ax2.set_xlabel('Image Feature PCs',size=15)
        ax2.set_ylabel('Technical factors',size=15)
        ax2.annotate('B', xycoords='axes fraction', xy=(0.8,0.95),size=20, color='green')
        fig.colorbar(im2, cax=cax2, orientation='vertical')


        import seaborn as sns
        sns.set_style("dark")
        import matplotlib as mpl
        label_size = 7
        mpl.rcParams['xtick.labelsize'] = label_size
        mpl.rcParams['ytick.labelsize'] = label_size
        for (k,l) in enumerate(['C', 'D', 'E', 'F', 'G']):
            [tf_name, pc_number, tf_vector, pc_vector, R, pv] = top5_scatter_results[k]
            axk = plt.subplot2grid((5,3), (k,2),rowspan=1,colspan=1)
            axk.scatter(tf_vector, pc_vector,s=2)
            axk.set_title('R: {:.2}, pv: {:.1}'.format(R, pv), size=10)
            axk.set_ylabel('PC {}'.format(pc_number))
            axk.set_xlabel(tf_name)
            axk.annotate(l, xycoords='axes fraction', xy=(0.8,0.7),size=20, color='green')


        plt.tight_layout()

        # plt.savefig(GTEx_directory + '/figures/associations/technical_factors_vs_pca_expression.eps',format='eps', dpi=100)

        os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)

        plt.show()


    @staticmethod
    def expression_PCs_vs_image_feature_PCs():

        [R_matrix, pv_matrix, top5_scatter_results] = pickle.load(open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'rb'))
        norm = MidPointNorm(midpoint=0)


        fig = plt.figure(figsize=(15,5))
        ax1 = plt.subplot2grid((2,5), (0,0),colspan=2, rowspan=1)
        ax2 = plt.subplot2grid((2,5), (0,3),colspan=2, rowspan=1)


        divider1 = make_axes_locatable(ax1)
        divider2 = make_axes_locatable(ax2)
        cax1 = divider1.append_axes('right', size='5%', pad=0.05)
        cax2 = divider2.append_axes('right', size='5%', pad=0.05)

        im1 = ax1.imshow(R_matrix[0:10,0:20], norm=norm, cmap=PL.get_cmap("coolwarm"))
        ax1.set_title("Pearson R values", size=12)
        ax1.set_xlabel('Image Feature PCs', size=12)
        ax1.set_ylabel('Expression PCs', size=12)
        ax1.annotate('A', xycoords='axes fraction', xy=(0.9,0.8),size=20, color='green')
        plt.colorbar(im1, cax=cax1, orientation='vertical')

        im2 = ax2.imshow(-np.log10(pv_matrix[0:10,0:20]),cmap=PL.get_cmap("Reds"))

        ax2.set_title("-log10 pvalues", size=12)
        ax2.set_xlabel('Image Feature PCs', size=12)
        ax2.set_ylabel('Expression PCs', size=12)
        ax2.annotate('B', xycoords='axes fraction', xy=(0.9,0.8),size=20, color='green')
        plt.colorbar(im2, cax=cax2, orientation='vertical')

        import seaborn as sns
        sns.set_style("dark")
        import matplotlib as mpl
        label_size = 7
        mpl.rcParams['xtick.labelsize'] = label_size
        mpl.rcParams['ytick.labelsize'] = label_size


        for (k,l) in enumerate(['C', 'D', 'E', 'F', 'G']):
            [expression_pc_number, image_feature_pc_number, expression_pc_vector, image_pc_vector, R, pv] = top5_scatter_results[k]
            axk = plt.subplot2grid((2,5), (1,k),rowspan=1,colspan=1)
            axk.scatter(expression_pc_vector, image_pc_vector,s=2)
            axk.set_title('R: {:.2}, pv: {:.1}'.format(R, pv), size=10)
            axk.set_ylabel('Expression PC {}'.format(expression_pc_number))
            axk.set_xlabel('Image feature PC {}'.format(image_feature_pc_number))
            axk.annotate(l, xycoords='axes fraction', xy=(0.85,0.8),size=20, color='green')

        plt.tight_layout()

        os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)

        plt.show()




if __name__ == '__main__':
    eval(group + '().' + name + '()')
