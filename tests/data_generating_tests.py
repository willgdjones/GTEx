import gzip
import pandas as pd
import numpy as np
import h5py
import os
import requests
import argparse

tissue_types = ['Lung','Artery - Tibial','Heart - Left Ventricle', 'Breast - Mammary Tissue', 'Brain - Cerebellum', 'Pancreas', 'Testis', 'Liver', 'Ovary', 'Stomach']

def test_generated_covering_patches():
    total_batches = []
    for tissue in tissue_types:
        IDs = os.listdir('data/processed/covering_patches/{}'.format(tissue))
        for ID in IDs:
            batches_length = len(os.listdir('data/processed/covering_patches/{}/{}'.format(tissue,ID)))
            total_batches.append(batches_length)
    print (sum(np.array(total_batches) == 0))
    print (len(total_batches))
    print (sum(np.array(total_batches) >= 1))
    assert all(np.array(total_batches) > 1)
            
