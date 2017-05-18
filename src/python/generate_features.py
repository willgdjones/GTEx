import sys
import os
import numpy as np
import h5py
import pdb
from scipy.misc import imresize
import matplotlib.pyplot as plt
from keras.layers import Input, GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
import argparse
import ipdb

sys.path = ['/nfs/gns/homes/willj/anaconda3/envs/GTEx/lib/python3.5/site-packages'] + sys.path
GTEx_directory = '/hps/nobackup/research/stegle/users/willj/GTEx'
patch_sizes = [128, 256, 512, 1024, 2048, 4096]

tissue_filepath = os.path.join(GTEx_directory,'data','raw',tissue)
tissue_images = os.listdir(tissue_filepath)
tissue_IDs = [x.split('.')[0] for x in tissue_images]

def retrained_inception_model():
    inception_model = InceptionV3(weights='imagenet', include_top=False)
    x = inception_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(10, activation='softmax')(x)

    model = Model(input=inception_model.input, output=predictions)
    model.load_weights(os.path.join(GTEx_directory, 'models','inception_50_-1.h5'))
    return model


def raw_inception_model():
    model = InceptionV3(weights='imagenet', include_top=False)
    return model

parser = argparse.ArgumentParser(description='')
parser.add_argument('-i','--ID', help='GTEx ID', required=True)
parser.add_argument('-m','--model_choice', help='Retrained / Raw Inceptionet', required=True)
args = vars(parser.parse_args())
ID = args['ID']
model_choice = args['model_choice']

assert model_choice == 'raw' or model_choice == 'retrained', 'Choose either raw or retrained model'


if model_choice == 'raw':
    model = raw_inception_model()
elif model_choic == 'retrained':
    model = retrained_inception_model()

print ('ID: {}, model: {}'.format(ID,aggregation))

for ps in patch_sizes:
    patch_path = GTEx_directory + '/data/patches/{tissue}/{tissue}_{patch_size}.hdf5'.format(tissue=tissue,ID=ID,patch_size=ps)
    g = h5py.File(patch_path, 'r')
    patches = g['patches']
    patches = np.array([imresize(x, (299,299)) for x in patches])
    print (len(patches))

    image_features = []
    upper_limit = int(len(patches)/100)
    for i in range(upper_limit + 1):
        print (i*100)
        batch_tiles = patches[i*100:(i+1)*100,:,:,:].astype(np.float16) / 255
        if len(batch_tiles) == 0:
            continue
        batch_features = inception_model.predict(batch_tiles)
        ipdb.set_trace()
        image_features.append(batch_features)

    image_features = np.array(image_features)
    image_features = np.vstack(image_features)


    if model_choice == 'retrained':
        assert np.array(image_features).shape[1] == 1024
    elif model_choice == 'raw':
        assert np.array(image_features).shape[1] == 2048

    mean_features = np.mean(image_features,axis=0)
    median_features = np.median(image_features,axis=0)
    max_features = np.max(image_features,axis=0)

    feature_path = GTEx_directory + '/data/features/{tissue}/{model_choice}_{ID}_{patch_size}.hdf5'.format(model_choice=model_choice,ID=ID,patch_size=ps)

    with h5py.File(feature_path,'w') as h:
        h.create_dataset('features', data=image_features)
        h.create_dataset('max', data=max_features)
        h.create_dataset('mean', data=mean_features)
        h.create_dataset('median', data=median_features)
