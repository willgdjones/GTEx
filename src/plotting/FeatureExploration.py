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
        ax[0].annotate('A', xycoords='axes fraction', xy=(0.1, 0.85), size=50, color='green')


        ax[1].imshow(slidemarkings[ylim[0]:ylim[1], xlim[0]:xlim[1], :])
        ax[1].axis('off')
        ax[1].annotate('B', xycoords='axes fraction', xy=(0.1, 0.85), size=50, color='green')

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
        M = 2000

        fig, ax = plt.subplots(1,2, figsize=(10,5))
        ax[0].hist(np.mean(expression,axis=0),bins=100)
        cutoff = k
        ax[0].plot([cutoff, cutoff], [0, 4500],c='red')
        # ax[0].set_title("Histogram of mean gene expression")
        ax[0].set_xlabel("Mean", size=30)
        ax[0].set_ylabel("Count", size=30)
        ax[0].tick_params(axis='both', labelsize=30)
        ax[0].annotate('A', xycoords='axes fraction', xy=(0.85,0.85),size=30, color='green')



        filt_expression_mean_idx = np.mean(expression,axis=0) > 1
        filt_expression = expression[:, filt_expression_mean_idx]

        plt.hist(np.std(filt_expression,axis=0),bins=100)

        # cutoff = min(np.mean(expression[:,np.argsort(np.mean(expression,axis=0))[-1000:]],axis=0))
        cutoff = min(np.std(expression[:,np.argsort(np.std(expression,axis=0))[-M:]],axis=0))
        ax[1].plot([cutoff, cutoff], [0, 2500],c='red')
        # ax[1].set_title("Histogram of gene expression standard deviation")
        ax[1].set_xlabel("Standard devation", size=30)
        ax[1].tick_params(axis='both', labelsize=30)
        ax[1].annotate('B', xycoords='axes fraction', xy=(0.85,0.85),size=30, color='green')

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
        ax[0].annotate('A', xycoords='axes fraction', xy=(0.93,0.76),size=20, color='green')
        im = ax[1].imshow(raw_image_features[0:100, 0:50].T.astype(np.float32), cmap='Reds')
        ax[1].set_ylabel("Raw Inceptionet")
        ax[1].set_xlabel("Samples")
        ax[1].annotate('B', xycoords='axes fraction', xy=(0.93,0.87),size=20, color='green')
        fig.colorbar(im, ax=ax.ravel().tolist())
        # ax[0].imshow(raw_image_features[:, 0:50].T.astype(np.float32), cmap='Reds')

        os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)

        plt.show()

    @staticmethod
    def features_across_patches():
        [features1, image1, features2, image2] = pickle.load(open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'rb'))


        fig = plt.figure(figsize=(15, 4))
        ax11 = plt.subplot2grid((1, 4), (0, 0), colspan=1, rowspan=1)
        ax12 = plt.subplot2grid((1, 4), (0, 1), colspan=1, rowspan=1)
        ax21 = plt.subplot2grid((1, 4), (0, 2), colspan=1, rowspan=1)
        ax22 = plt.subplot2grid((1, 4), (0, 3), colspan=1, rowspan=1)

        divider1 = make_axes_locatable(ax11)
        divider2 = make_axes_locatable(ax21)
        cax1 = divider1.append_axes('right', size='2%', pad=0.05)
        cax2 = divider2.append_axes('right', size='2%', pad=0.05)

        ax11.set_title("GTEX-117YW-0526", size=20)
        im1 = ax11.imshow(features1.T[0:50,0:100].astype(np.float32), cmap='Reds')
        ax11.set_xlabel('Patches', size=20)
        ax11.set_ylabel('Image feature', size=20)
        fig.colorbar(im1, cax=cax1, orientation='vertical')
        ax12.set_title("GTEX-117YW-0526", size=20)
        ax12.imshow(image1)
        ax12.axis('off')


        ax21.set_title("GTEX-117YX-1326", size=20)
        im2 = ax21.imshow(features2.T[0:50,0:100].astype(np.float32), cmap='Reds')
        ax21.set_xlabel('Patches', size=20)
        # ax21.set_ylabel('Image feature', size=20)
        fig.colorbar(im2, cax=cax2, orientation='vertical')
        ax22.set_title("GTEX-117YX-1326", size=20)
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

    @staticmethod
    def top5_bottom5_feature796():

        results = pickle.load(open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'rb'))
        top_five_images, bottom_five_images = results

        fig, ax = plt.subplots(2,5, figsize=(20,6))

        for i in range(5):
            ax[0][i].imshow(top_five_images[i])
            ax[0][i].axis('off')

        for i in range(5):
            ax[1][i].imshow(bottom_five_images[i])
            ax[1][i].axis('off')

        os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)

        plt.show()

    @staticmethod
    def top5_bottom5_feature671():

        results = pickle.load(open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'rb'))
        top_five_images, bottom_five_images = results

        fig, ax = plt.subplots(2,5, figsize=(20,6))

        for i in range(5):
            ax[0][i].imshow(top_five_images[i])
            ax[0][i].axis('off')

        for i in range(5):
            ax[1][i].imshow(bottom_five_images[i])
            ax[1][i].axis('off')

        os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)
        plt.show()

    @staticmethod
    def patches_at_different_scales():
        patch = pickle.load(open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'rb'))
        fig, ax = plt.subplots(1,6, figsize=(10,3))

        SIZES = [128, 256, 512, 1024, 2048, 4096]
        for (i, s) in enumerate(SIZES):
            ax[i].set_title(s)
            ax[i].imshow(patch[s])
            ax[i].axis('off')

        os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)

        plt.show()


    @staticmethod
    def image_feature_gradient671():

        thumbnails = pickle.load(open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'rb'))
        f,a = plt.subplots(19,15,figsize=(30,30))
        for (i, th) in enumerate(thumbnails):
            a.flatten()[i].imshow(th[0])
            a.flatten()[i].set_title(th[1], size=5)
            a.flatten()[i].axis('off')
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)
        plt.show()

    @staticmethod
    def image_feature_gradient796():

        thumbnails = pickle.load(open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'rb'))
        f,a = plt.subplots(19,15,figsize=(30,30))
        for (i, th) in enumerate(thumbnails):
            a.flatten()[i].imshow(th[0])
            a.flatten()[i].set_title(th[1], size=5)
            a.flatten()[i].axis('off')
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)
        plt.show()

    @staticmethod
    def image_feature_gradient211():

        thumbnails = pickle.load(open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'rb'))
        f,a = plt.subplots(19,15,figsize=(30,30))
        for (i, th) in enumerate(thumbnails):
            a.flatten()[i].imshow(th[0])
            a.flatten()[i].set_title(th[1], size=5)
            a.flatten()[i].axis('off')
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)
        plt.show()

    @staticmethod
    def image_feature_gradient501():

        thumbnails = pickle.load(open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'rb'))
        f,a = plt.subplots(19,15,figsize=(30,30))
        for (i, th) in enumerate(thumbnails):
            a.flatten()[i].imshow(th[0])
            a.flatten()[i].set_title(th[1], size=5)
            a.flatten()[i].axis('off')
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)
        plt.show()

        import pdb; pdb.set_trace()





if __name__ == '__main__':
    eval(group + '().' + name + '()')
