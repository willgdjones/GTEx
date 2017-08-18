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
        import seaborn as sns
        sns.set_style("dark")

        validation_accuracies = np.loadtxt(GTEx_directory + '/results/{group}/{name}.txt'.format(group=group, name=name))
        plt.plot(validation_accuracies)
        plt.title("Top validation accuracy vs patchsize", size=20)
        plt.ylabel('Validation accuracy', size=15)
        plt.xticks([0, 1, 2, 3, 4, 5], ['128', '256', '512', '1024', '2056', '4096'])
        plt.xlabel('Patch size', size=15)

        os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)
        plt.show()

class PCAFeatureAssociations:

    @staticmethod
    def expression_PCs_vs_TFs():

        [R_matrix, pv_matrix, top5_scatter_results] = pickle.load(open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group,name=name), 'rb'))
        norm = MidPointNorm(midpoint=0)


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
        plt.colorbar(im1, cax=cax1, orientation='vertical')


        im2 = ax2.imshow(-np.log10(pv_matrix[:,0:10]),cmap=PL.get_cmap("Reds"))

        ax2.set_title("-log10 pvalues", size=15)
        ax2.set_xlabel('Expression PCs', size=15)
        ax2.set_ylabel('Technical factors', size=15)
        plt.colorbar(im2, cax=cax2, orientation='vertical')

        import seaborn as sns
        sns.set_style("dark")
        import matplotlib as mpl
        label_size = 7
        mpl.rcParams['xtick.labelsize'] = label_size
        mpl.rcParams['ytick.labelsize'] = label_size


        for k in range(5):
            [tf_name, pc_number, tf_vector, pc_vector, R, pv] = top5_scatter_results[k]
            axk = plt.subplot2grid((5,3), (k,2),rowspan=1,colspan=1)
            axk.scatter(tf_vector, pc_vector,s=2)
            axk.set_title('R: {:.2}, pv: {:.1}'.format(R, pv), size=10)
            axk.set_ylabel('PC {}'.format(pc_number))
            axk.set_xlabel(tf_name)

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


        divider1 = make_axes_locatable(ax1)
        divider2 = make_axes_locatable(ax2)
        cax1 = divider1.append_axes('right', size='5%', pad=0.05)
        cax2 = divider2.append_axes('right', size='5%', pad=0.05)


        im1 = ax1.imshow(R_matrix[:,0:10],norm=norm,cmap=PL.get_cmap("coolwarm"))
        ax1.set_title("Pearson R values",size=15)
        ax1.set_xlabel('Expression PCs',size=15)
        ax1.set_ylabel('Technical factors',size=15)
        fig.colorbar(im1,cax=cax1, orientation='vertical')


        im2 = ax2.imshow(-np.log10(pv_matrix[:,0:10]),cmap=PL.get_cmap("Reds"))
        ax2.set_title("-log10 pvalues",size=15)
        ax2.set_xlabel('Image Feature PCs',size=15)
        ax2.set_ylabel('Technical factors',size=15)
        fig.colorbar(im2, cax=cax2, orientation='vertical')


        import seaborn as sns
        sns.set_style("dark")
        import matplotlib as mpl
        label_size = 7
        mpl.rcParams['xtick.labelsize'] = label_size
        mpl.rcParams['ytick.labelsize'] = label_size
        for k in range(5):
            [tf_name, pc_number, tf_vector, pc_vector, R, pv] = top5_scatter_results[k]
            axk = plt.subplot2grid((5,3), (k,2),rowspan=1,colspan=1)
            axk.scatter(tf_vector, pc_vector,s=2)
            axk.set_title('R: {:.2}, pv: {:.1}'.format(R, pv), size=10)
            axk.set_ylabel('PC {}'.format(pc_number))
            axk.set_xlabel(tf_name)


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
        plt.colorbar(im1, cax=cax1, orientation='vertical')

        im2 = ax2.imshow(-np.log10(pv_matrix[0:10,0:20]),cmap=PL.get_cmap("Reds"))

        ax2.set_title("-log10 pvalues", size=12)
        ax2.set_xlabel('Image Feature PCs', size=12)
        ax2.set_ylabel('Expression PCs', size=12)
        plt.colorbar(im2, cax=cax2, orientation='vertical')

        import seaborn as sns
        sns.set_style("dark")
        import matplotlib as mpl
        label_size = 7
        mpl.rcParams['xtick.labelsize'] = label_size
        mpl.rcParams['ytick.labelsize'] = label_size


        for k in range(5):
            [expression_pc_number, image_feature_pc_number, expression_pc_vector, image_pc_vector, R, pv] = top5_scatter_results[k]
            axk = plt.subplot2grid((2,5), (1,k),rowspan=1,colspan=1)
            axk.scatter(expression_pc_vector, image_pc_vector,s=2)
            axk.set_title('R: {:.2}, pv: {:.1}'.format(R, pv), size=10)
            axk.set_ylabel('Expression PC {}'.format(expression_pc_number))
            axk.set_xlabel('Image feature PC {}'.format(image_feature_pc_number))

        plt.tight_layout()

        os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)

        plt.show()


class FeatureExploration():
    @staticmethod
    def extracting_tissue_patches():

        [slide, mask, slidemarkings] = pickle.load(open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'rb'))

        fig, ax = plt.subplots(1, 2, figsize=(20,10))

        ylim = (500, 1300)
        xlim = (250, 1100)
        zoom_slide = slide[ylim[0]:ylim[1], xlim[0]:xlim[1], :]
        zoom_mask = mask[ylim[0]:ylim[1], xlim[0]:xlim[1]]

        ax[0].imshow(cv2.bitwise_and(zoom_slide, zoom_slide, mask=zoom_mask))
        ax[0].axis('off')

        ax[1].imshow(slidemarkings[ylim[0]:ylim[1], xlim[0]:xlim[1], :])
        ax[1].axis('off')

        os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)

        plt.show()


    @staticmethod
    def feature_variation_across_patchsize():
        import seaborn as sns
        sns.set_style("dark")


        patch_sizes = [128, 256, 512, 1024, 2048, 4096]

        all_image_features = pickle.load(open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'rb'))

        f, a = plt.subplots(1,6, figsize=(35,5))
        # f.suptitle("Image feature variation. Lung, patch-size 256",size=30)
        for (i,s) in enumerate(patch_sizes):
            a[i].hist(np.std(all_image_features[s],axis=0),bins=100)
            a[i].set_title("Patch-size {}".format(s),size=20)
        plt.tight_layout()
        plt.subplots_adjust(top=0.80)

        os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)

        plt.show()

    @staticmethod
    def feature_variation_concatenated():
        import seaborn as sns
        sns.set_style("dark")

        patch_sizes = [128, 256, 512, 1024, 2048, 4096]

        all_image_features = pickle.load(open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'rb'))

        plt.figure()
        concatenated_features = np.vstack([all_image_features[s] for s in patch_sizes])
        plt.hist(np.std(concatenated_features,axis=0),bins=100)
        cutoff = min(np.std(concatenated_features[:,np.argsort(np.std(concatenated_features,axis=0))[-500:]],axis=0))
        plt.plot([cutoff, cutoff], [0, 300],c='red')
        plt.title("Histogram of variance from concatenated features across patch-sizes",size=11)
        plt.xlabel("Variance")
        plt.ylabel("Counts")
        plt.tight_layout()


        os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)

        plt.show()


    @staticmethod
    def expression_means_and_stds():
        import seaborn as sns
        sns.set_style("dark")

        [expression, transcriptIDs] = pickle.load(open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'rb'))

        k = 1
        M = 5000

        fig, ax = plt.subplots(1,2, figsize=(10,5))
        ax[0].hist(np.mean(expression,axis=0),bins=100)
        cutoff = k
        ax[0].plot([cutoff, cutoff], [0, 4500],c='red')
        # ax[0].set_title("Histogram of mean gene expression")
        ax[0].set_xlabel("Mean expression")
        ax[0].set_ylabel("Count")

        filt_expression_mean_idx = np.mean(expression,axis=0) > 1
        filt_expression = expression[:, filt_expression_mean_idx]

        plt.hist(np.std(filt_expression,axis=0),bins=100)

        # cutoff = min(np.mean(expression[:,np.argsort(np.mean(expression,axis=0))[-1000:]],axis=0))
        cutoff = min(np.std(expression[:,np.argsort(np.std(expression,axis=0))[-5000:]],axis=0))
        ax[1].plot([cutoff, cutoff], [0, 2500],c='red')
        # ax[1].set_title("Histogram of gene expression standard deviation")
        ax[1].set_xlabel("Expression standard devation")
        ax[1].set_ylabel("Count")

        plt.tight_layout()

        os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)

        plt.show()

    @staticmethod
    def aggregated_features_across_samples():
        import matplotlib as mpl
        [retrained_image_features, raw_image_features] = pickle.load(open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'rb'))

        fig, ax = plt.subplots(2,1, figsize=(10, 10))
        im = ax[0].imshow(retrained_image_features[0:100, 0:50].T.astype(np.float32), cmap='Reds')
        ax[0].set_ylabel("Retrained Inceptionet")
        ax[0].set_xlabel("Samples")
        im = ax[1].imshow(raw_image_features[0:100, 0:50].T.astype(np.float32), cmap='Reds')
        ax[1].set_ylabel("Raw Inceptionet")
        ax[1].set_xlabel("Samples")
        fig.colorbar(im, ax=ax.ravel().tolist())
        # ax[0].imshow(raw_image_features[:, 0:50].T.astype(np.float32), cmap='Reds')

        os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)

        plt.show()

    @staticmethod
    def features_across_patches():
        [features1, image1, features2, image2] = pickle.load(open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'rb'))


        fig = plt.figure(figsize=(12, 10))
        ax11 = plt.subplot2grid((2, 2), (0, 0), colspan=1, rowspan=1)
        ax12 = plt.subplot2grid((2, 2), (0, 1), colspan=1, rowspan=1)
        ax21 = plt.subplot2grid((2, 2), (1, 0), colspan=1, rowspan=1)
        ax22 = plt.subplot2grid((2, 2), (1, 1), colspan=1, rowspan=1)

        divider1 = make_axes_locatable(ax11)
        divider2 = make_axes_locatable(ax21)
        cax1 = divider1.append_axes('right', size='2%', pad=0.05)
        cax2 = divider2.append_axes('right', size='2%', pad=0.05)

        ax11.set_title("Features across patches. Lung GTEX-117YW-0526")
        im1 = ax11.imshow(features1.T[0:50,0:100].astype(np.float32), cmap='Reds')
        ax11.set_xlabel('Patch', size=15)
        ax11.set_ylabel('Image feature', size=15)
        fig.colorbar(im1, cax=cax1, orientation='vertical')
        ax12.set_title("Lung GTEX-117YW-0526 image")
        ax12.imshow(image1)
        ax12.axis('off')


        ax21.set_title("Features across patches. Lung GTEX-117YX-1326")
        im2 = ax21.imshow(features2.T[0:50,0:100].astype(np.float32), cmap='Reds')
        ax21.set_xlabel('Patch', size=15)
        ax21.set_ylabel('Image feature', size=15)
        fig.colorbar(im2, cax=cax2, orientation='vertical')
        ax22.set_title("Lung GTEX-117YX-1326 image")
        ax22.imshow(image2)
        ax22.axis('off')

        os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)

        plt.show()

    @staticmethod
    def feature_crosscorrelation():
        import pylab
        import scipy.cluster.hierarchy as sch

        D = pickle.load(open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'rb'))

        Dc = D.copy()
        # Compute and plot first dendrogram.
        fig = pylab.figure(figsize=(5,5))
        ax1 = fig.add_axes([0.09,0.1,0.2,0.6])
        Y = sch.linkage(Dc, method='centroid')
        Z1 = sch.dendrogram(Y, orientation='right')
        # Z1 = sch.dendrogram(Y, orientation='left')
        ax1.set_xticks([])
        ax1.set_yticks([])

        axmatrix = fig.add_axes([0.3,0.1,0.6,0.6])
        idx1 = Z1['leaves']
        # idx2 = Z2['leaves']
        Dc = Dc[idx1,:]
        Dc = Dc[:,idx1]
        # D = D[:,idx2]
        im = axmatrix.matshow(Dc, aspect='auto', origin='lower', cmap=pylab.cm.YlGnBu)
        axmatrix.set_xticks([])
        axmatrix.set_yticks([])

        # Plot colorbar.
        axcolor = fig.add_axes([0.91,0.1,0.02,0.6])
        pylab.colorbar(im, cax=axcolor)

        os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)

        plt.show()







class RawFeatureAssociations():

    @staticmethod
    def raw_associations_across_patchsizes():
        import seaborn as sns
        sns.set_style("dark")

        SIZES = [128, 256, 512, 1024, 2048, 4096]
        ALPHAS = [0.01, 0.0001, 0.000001]

        all_counts = pickle.load(open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'rb'))

        plt.figure(figsize=(10,7))
        plt.title("Number of significant p-values (Bonf), varying patch size and FDR", size=17)
        plt.xticks(range(len(SIZES)), SIZES,size=15)
        plt.xlabel('Patch size', size=15)
        plt.ylabel('Number of significant pvalues (Bonf)', size=15)
        colours = ['blue','red','green']
        for (k, alph) in enumerate(ALPHAS):
            plt.plot(all_counts[k], c=colours[k],label=alph)

        plt.legend()
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

        plt.figure(figsize=(10,7))
        plt.title("Number of significant p-values. Raw vs Retrained Inceptionet",size=20)
        plt.xticks(range(len(SIZES)), SIZES, size=15)
        plt.xlabel('Patch size', size=15)
        plt.ylabel('Number of significant pvalues (Bonf)', size=15)
        colours = ['blue','red']
        for (k, m) in enumerate(MODELS):
            plt.plot(all_counts[k], c=colours[k],label=m)

        plt.legend()
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

        plt.figure(figsize=(10,7))
        plt.title("Comparing Aggregation methods",size=20)
        plt.xticks(range(len(SIZES)), SIZES, size=15)
        plt.xlabel('Patch size', size=15)
        plt.ylabel('Number of significant pvalues (Bonf)', size=15)
        colours = ['blue','red']
        for (k, m) in enumerate(AGGREGATIONS):
            plt.plot(all_counts[k], c=colours[k],label=m)

        plt.legend()
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
        plt.figure(figsize=(10,10))
        plt.title("Number of features with significant pvalues (Bonf)", size=20)
        plt.xticks(range(len(SIZES)),SIZES,size=15)
        plt.xlabel('Patch size',size=15)
        plt.ylabel('Number of features with significant pvalues (Bonf)',size=15)
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
        plt.figure(figsize=(10,10))
        plt.title("Number of transcripts significant to at least 1 feature", size=20)
        plt.xticks(range(len(SIZES)),SIZES,size=15)
        plt.xlabel('Patch size',size=15)
        plt.ylabel('Number of features with significant pvalues (Bonf)',size=15)
        plt.plot(size_counts)

        os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)

        plt.show()


class Progress():


    @staticmethod
    def gantt_chart():
        import plotly
        import plotly.plotly as py
        plotly.offline.init_notebook_mode()
        plotly.tools.set_credentials_file(username='willgdjones', api_key='EPtOoWzAHN5hLaZElyoQ')

        from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
        import plotly.figure_factory as ff



        df = [dict(Task="Job A", Start='2009-01-01', Finish='2009-02-28'),
              dict(Task="Job B", Start='2009-03-05', Finish='2009-04-15'),
              dict(Task="Job C", Start='2009-02-20', Finish='2009-05-30')]

        fig = ff.create_gantt(df)

        plot(fig, filename= GTEx_directory + '/plotting/{group}/{name}.html'.format(group=group, name=name))

        # os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)












if __name__ == '__main__':
    eval(group + '().' + name + '()')
