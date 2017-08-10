import argparse
GTEx_directory = '/hps/nobackup/research/stegle/users/willj/GTEx'
import os
import sys
sys.path.append(GTEx_directory)
import h5py
import gzip
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import pylab as PL
from src.utils.helpers import *

tissues = ['Artery - Tibial',
        'Brain - Cerebellum',
            'Breast - Mammary Tissue',
            'Heart - Left Ventricle',
            'Liver',
            'Lung',
            'Ovary',
            'Pancreas',
            'Stomach',
            'Testis']

parser = argparse.ArgumentParser(description='')
parser.add_argument('-p', '--percentile', help='Percentage of patches to include', required=True)
parser.add_argument('-s', '--patchsize', help='Specify patchsize', required=True)
args = vars(parser.parse_args())
patchsize = args['patchsize']
pc = int(args['percentile'])

def generate_across_tissue_patch_set(patchsize, pc):
    all_patches = []
    all_features = []
    all_labels = []
    with h5py.File(os.path.join(GTEx_directory,'data/h5py/collected_features.h5py'),'r') as f:
        for tissue in tissues:
            IDlist = list(f[tissue]['-1'][patchsize]['retrained'])
            print (tissue)
            for (k,ID) in enumerate(IDlist):
                if k % 1 == 0:
                    print ('{}/{}'.format(k,len(IDlist)))
                with h5py.File(os.path.join(GTEx_directory,'data/patches/{}/{}_{}.hdf5'.format(tissue,ID,patchsize)),'r') as g:
                    patches = g['patches'].value
                features = f[tissue]['-1'][patchsize]['retrained'][ID]['features'].value
                assert patches.shape[0] == features.shape[0]
                n = patches.shape[0]
                idx = np.random.choice(list(range(n)), int(n * (pc/100)))
                patches = patches[idx,:,:,:]
                features = features[idx,:]
                labels = [tissue] * features.shape[0]
                all_features.extend(features)
                all_patches.extend(patches)
                all_labels.extend(labels)
        all_patches = np.array(all_patches)
        all_features = np.array(all_features)

    with h5py.File(os.path.join(GTEx_directory,'data/h5py/across_patches_ps{}_pc{}.h5py'.format(patchsize,pc)),'w') as h:
        h.create_dataset('features',data=all_features)
        h.create_dataset('patches',data=all_patches)


generate_across_tissue_patch_set(patchsize, pc)
