import gzip
import pandas as pd
import numpy as np
import h5py
import os
import requests
import argparse
import pickle
import pdb

tissue_types = ['Lung','Artery - Tibial','Heart - Left Ventricle', 'Breast - Mammary Tissue', 'Brain - Cerebellum', 'Pancreas', 'Testis', 'Liver', 'Ovary', 'Stomach']

def test_representations():
    # for tile_size in ['small','medium','large']:
    for tile_size in ['small']:
        for tissue in tissue_types:
            reps_dir = 'data/processed/assembled_representations/inception_50_-1.h5/{}/{}'.format(tile_size, tissue) 
            try:
                for ID in np.random.choice(os.listdir(reps_dir),3):
                    reps = pickle.load(open(os.path.join(reps_dir, ID), 'rb'))
                    assert reps.shape[1] == 1024
            except ValueError:
                print ("Does not exist")
                continue

def test_association_data():
    # for tile_size in ['small','medium','large']:
    for tile_size in ['small']:
        for tissue in tissue_types:
            # print (tissue)
            assoc_dir = 'data/processed/association_data/expression/inception_50_-1.h5/{}/{}'.format(tile_size,tissue)
            # print (os.path.join(assoc_dir, 'X_y_mean'))
            data = pickle.load(open(os.path.join(assoc_dir, 'X_y_mean') ,'rb'))
            assert data[0].shape[0] == data[1].shape[0]


