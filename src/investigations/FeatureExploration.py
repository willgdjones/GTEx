import os
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import h5py
import argparse
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from matplotlib.colors import Normalize

from utils.helpers import *


GTEx_directory = '/hps/nobackup/research/stegle/users/willj/GTEx'

parser = argparse.ArgumentParser(description='Collection of experiments. Runs on the cluster.')
parser.add_argument('-g', '--group', help='Experiment group', required=True)
parser.add_argument('-n', '--name', help='Experiment name', required=True)
args = vars(parser.parse_args())
group = args['group']
name = args['name']




class FeatureExploration():

    @staticmethod
    def extracting_tissue_patches():
        os.makedirs(GTEx_directory + '/results/{}'.format(group), exist_ok=True)
        ID = 'GTEX-13FH7-1726'
        tissue = 'Lung'
        patchsize = 512
        slide, mask, slidemarkings = create_tissue_boundary(ID, tissue, patchsize)

        results = [slide, mask, slidemarkings]

        pickle.dump(results, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))


    @staticmethod
    def feature_variation_across_patchsize():
        os.makedirs(GTEx_directory + '/results/{}'.format(group), exist_ok=True)
        patch_sizes = [128, 256, 512, 1024, 2048, 4096]

        all_image_features = {}
        for ps in patch_sizes:
            image_features, _, _, _, _, _, _ = extract_final_layer_data('Lung', 'retrained', 'median', str(ps))
            all_image_features[ps] = image_features

        results = all_image_features


        pickle.dump(results, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))


    @staticmethod
    def feature_variation_concatenated():

        os.makedirs(GTEx_directory + '/results/{}'.format(group), exist_ok=True)
        patch_sizes = [128, 256, 512, 1024, 2048, 4096]

        all_image_features = {}
        for ps in patch_sizes:
            image_features, _, _, _, _, _, _ = extract_final_layer_data('Lung', 'retrained', 'median', str(ps))
            all_image_features[ps] = image_features

        results = all_image_features

        pickle.dump(results, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))

    @staticmethod
    def expression_means_and_stds():
        _, expression, _, transcriptIDs, _, _, _ = extract_final_layer_data('Lung', 'retrained', 'median', '256')

        results = [expression, transcriptIDs]
        pickle.dump(results, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))



    @staticmethod
    def aggregated_features_across_samples():
        retrained_image_features, _, _, _, _, _, _ = extract_final_layer_data('Lung', 'retrained', 'median', '256')
        raw_image_features, _, _, _, _, _, _ = extract_final_layer_data('Lung', 'raw', 'median', '256')
        results = [retrained_image_features, raw_image_features]
        pickle.dump(results, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))


    @staticmethod
    def features_across_patches():
        from openslide import open_slide
        os.makedirs(GTEx_directory + '/results/{}'.format(group), exist_ok=True)
        ID1 = 'GTEX-117YW-0526'
        ID2 = 'GTEX-117YX-1326'
        with h5py.File(os.path.join(GTEx_directory, 'data/h5py/collected_features.h5py'), 'r') as f:
            IDlist = list(f['Lung']['-1']['256']['retrained'])
            features1 = f['Lung']['-1']['256']['retrained'][ID1]['features'].value
            features2 = f['Lung']['-1']['256']['retrained'][ID2]['features'].value

        slide1 = open_slide(GTEx_directory + '/data/raw/Lung/{}.svs'.format(ID1))
        image1 = slide1.get_thumbnail(size=(800, 800))

        slide2 = open_slide(GTEx_directory + '/data/raw/Lung/{}.svs'.format(ID2))
        image2 = slide2.get_thumbnail(size=(800, 800))

        results = [features1, image1, features2, image2]

        pickle.dump(results, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))


    @staticmethod
    def feature_crosscorrelation():

        # Filter only non-zero features

        image_features, _, _, _, _, _, _ = extract_final_layer_data('Lung', 'retrained', 'median', '256')

        non_zero_idx = np.std(image_features,axis=0) > 0
        filt_image_features = image_features[:, non_zero_idx]

        def distance(x, y):
            dist = 1 - np.absolute(pearsonr(x,y)[0])
            return dist

        N = filt_image_features.shape[1]

        print ("Calculating features cross correlations")
        D = np.zeros([N,N])
        for i in range(N):
            if i % 100 == 0:
                print (i, '/{}'.format(N))
            for j in range(N):
                dist = distance(filt_image_features[:,i], filt_image_features[:,j])
                if np.isnan(dist) or dist > 1 or dist < 0:
                    import pdb; pdb.set_trace()
                D[i,j] = dist


        pickle.dump(D, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))


    @staticmethod
    def top5_bottom5_feature796():
        results = top5_bottom5_image('Lung', 'retrained', '256', 796)
        pickle.dump(results, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))

    @staticmethod
    def top5_bottom5_feature671():
        results = top5_bottom5_image('Lung', 'retrained', '256', 671)
        pickle.dump(results, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))

    @staticmethod
    def patches_at_different_scales():

        ID = 'GTEX-117YW-0526'
        patches = {}
        SIZES = [128, 256, 512, 1024, 2048, 4096]
        for ps in SIZES:
            with h5py.File(os.path.join(GTEx_directory, 'data/patches/Lung/{}_{}.hdf5'.format(ID, ps)), 'r') as f:
                patches[ps] = f['patches'].value


        patch = {}

        patch[128]= patches[128][100,:,:,:]
        patch[256]= patches[256][100,:,:,:]
        patch[512]= patches[512][100,:,:,:]
        patch[1024]= patches[1024][100,:,:,:]
        patch[2048]= patches[2048][80,:,:,:]
        patch[4096]= patches[4096][10,:,:,:]

        pickle.dump(patch, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))




if __name__ == '__main__':
    eval(group + '().' + name + '()')
