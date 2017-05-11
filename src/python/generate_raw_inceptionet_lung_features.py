import sys
import os
import numpy as np
import h5py
import pdb
from scipy.misc import imresize
import matplotlib.pyplot as plt

# patch_sizes = [4096]
patch_sizes = [128, 256, 512, 1024, 2048]

sys.path = ['/nfs/gns/homes/willj/anaconda3/envs/GTEx/lib/python3.5/site-packages'] + sys.path
GTEx_directory = '/hps/nobackup/research/stegle/users/willj/GTEx'

from keras.layers import Input, GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
import argparse

parser = argparse.ArgumentParser(description='')

parser.add_argument('-i','--ID', help='GTEx ID', required=True)
args = vars(parser.parse_args())
ID = args['ID']


def build_empty_model():
    inceptionet = InceptionV3(weights='imagenet', include_top=True)
    model = Model(inceptionet.input, inceptionet.layers[-2].output)
    return model

len(os.listdir(os.path.join(GTEx_directory, 'data/better_covering_patches')))

lung_filepath = os.path.join(GTEx_directory,'data','raw','Lung')
lung_images = os.listdir(lung_filepath)

lung_IDs = [x.split('.')[0] for x in lung_images]


model = build_empty_model()

final_layer_model = Model(model.input, model.layers[-2].output)


print (ID)

for ps in patch_sizes:
    # size = features[str(ps)]
    patch_path = GTEx_directory + '/data/better_covering_patches/{}_{}.hdf5'.format(ID,ps)
    g = h5py.File(patch_path, 'r')
    patches = g['patches']
    print (len(patches))
    patches = np.array([imresize(x, (299,299)) for x in patches])

    image_features = []
    upper_limit = int(len(patches)/100)
    for i in range(upper_limit + 1):
        print (i*100)
        batch_tiles = patches[i*100:(i+1)*100,:,:,:].astype(np.float16) / 255
        if len(batch_tiles) == 0:
            continue
        batch_features = final_layer_model.predict(batch_tiles)
        image_features.append(batch_features)

    image_features = np.array(image_features)
    image_features = np.vstack(image_features)
    image_features = np.squeeze(image_features)
    assert np.array(image_features).shape[1] == 2048
    feature_path = GTEx_directory + '/data/raw_inceptionet_features/{}_{}.hdf5'.format(ID,ps)
    with h5py.File(feature_path,'w') as h:
        h.create_dataset(ID, data=image_features)
