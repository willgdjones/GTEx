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
import pdb

parser = argparse.ArgumentParser(description='')
parser.add_argument('-t', '--tissue', help='Tissue', required=True)
parser.add_argument('-p', '--percentile', help='Percentage of patches to include', required=True)
parser.add_argument('-s','--patchsize', help='Specify patchsize', required=True)
args = vars(parser.parse_args())
tissue = args['tissue']
patchsize = args['patchsize']
pc = int(args['percentile'])



def generate_within_tissue_patch_set(tissue, patchsize, pc):

    all_patches = []
    all_features = []
    with h5py.File(os.path.join(GTEx_directory,'data/h5py/collected_features.h5py'),'r') as f:
        IDlist = list(f[tissue]['-1'][patchsize]['retrained'])
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
            all_features.extend(features)
            all_patches.extend(patches)
        all_patches = np.array(all_patches)
        all_features = np.array(all_features)
    with h5py.File(os.path.join(GTEx_directory,'data/h5py/within_{}_patches_ps{}_pc{}.h5py'.format(tissue,patchsize,pc)),'w') as h:
        h.create_dataset('features',data=all_features)
        h.create_dataset('patches',data=all_patches)


generate_within_tissue_patch_set(tissue, patchsize, pc)
