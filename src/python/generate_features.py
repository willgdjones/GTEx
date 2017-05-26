import sys
import os
import numpy as np
import h5py
from scipy.misc import imresize
import matplotlib.pyplot as plt
from keras.layers import Input, GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
import argparse
import glob
import pdb
sys.path.append('/nfs/gns/homes/willj/anaconda3/envs/GTEx/lib/python3.5/site-packages')

GTEx_directory = '/hps/nobackup/research/stegle/users/willj/GTEx'
layers = [1, 7, 12, 65, 166, -1]
patch_sizes = [128, 256, 512, 1024, 2048, 4096]
tissue_types = ['Lung', 'Artery - Tibial', 'Heart - Left Ventricle', 'Breast - Mammary Tissue', 'Brain - Cerebellum', 'Pancreas', 'Testis', 'Liver', 'Ovary', 'Stomach']
model_choices = ['raw','retrained']
convolutional_aggregations = ['mean','max']



def generate_retrained_inception_models():
    inception_model = InceptionV3(weights='imagenet', include_top=False)
    x = inception_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(10, activation='softmax')(x)

    prediction_model = Model(input=inception_model.input, output=predictions)
    weights_file = os.path.join(GTEx_directory, 'models', 'inception_50_-1.h5')

    prediction_model.load_weights(weights_file)
    retrained_feature_model = Model(input=prediction_model.input,output=prediction_model.layers[-2].output)
    retrained_feature_models = dict([(l,Model(input=retrained_feature_model.input, output=retrained_feature_model.layers[l].output)) for l in layers])
    return retrained_feature_models

def generate_raw_inception_models():
    raw_prediction_model = InceptionV3(weights='imagenet', include_top=True)

    raw_feature_model = Model(input=raw_prediction_model.input, output=raw_prediction_model.layers[-2].output)
    raw_feature_models = dict([(l,Model(input=raw_feature_model.input, output=raw_feature_model.layers[l].output)) for l in layers])
    return raw_feature_models 

parser = argparse.ArgumentParser(description='')
parser.add_argument('-p','--pair', help='GTEx ID, Tissue pair', required=True)
args = vars(parser.parse_args())
pair = args['pair'].split(' ')
tissue = ' '.join(pair[1:])
ID = pair[0]
os.makedirs( GTEx_directory + '/data/features/{}'.format(tissue), exist_ok=True)

tissue_filepath = os.path.join(GTEx_directory, 'data', 'raw', tissue)
tissue_images = os.listdir(tissue_filepath)
tissue_IDs = [x.split('.')[0] for x in tissue_images]

raw_inception_models = generate_raw_inception_models()
retrained_inception_models = generate_retrained_inception_models()

print ('ID: {}, Tissue: {}'.format(ID,tissue))

for ps in patch_sizes:
    print ("\nPatch size: {}".format(ps))
    patch_path = GTEx_directory + '/data/patches/{tissue}/{ID}_{patch_size}.hdf5'.format(tissue=tissue,ID=ID,patch_size=ps)
    try:
        g = h5py.File(patch_path, 'r')
    except:
        continue
    print ("Loading patches")

    patches = g['patches']
    patches = np.array([imresize(x, (299,299)) for x in patches])
    num_patches = len(patches)
    print ('Number of patches: {}'.format(num_patches))

    for model_choice in model_choices:
        print ('\nModel: {}'.format(model_choice))
        layer_models = eval('{}_inception_models'.format(model_choice))

        for l in layers:
            print ('Layer: {}'.format(l))
            model_features = []

            model = layer_models[l]
            if len(glob.glob(GTEx_directory + '/data/features/{tissue}/{model_choice}_{ID}_{patch_size}_l{layer}_*.hdf5'.format(tissue=tissue,model_choice=model_choice,ID=ID,layer=l,patch_size=ps))) > 0:
                print ('Features already exist')
                continue
            elif len(glob.glob(GTEx_directory + '/data/features/{tissue}/{model_choice}_{ID}_{patch_size}_l{layer}.hdf5'.format(tissue=tissue,model_choice=model_choice,ID=ID,layer=l,patch_size=ps))) > 0:
                print ('Features already exist')
                continue

            upper_limit = int(len(patches)/100)
            for i in range(upper_limit + 1):
                print ('batch {}'.format(i*100))
                batch_tiles = patches[i*100:(i+1)*100,:,:,:].astype(np.float16) / 255
                if len(batch_tiles) == 0:
                    continue
                batch_features = model.predict(batch_tiles).astype(np.float16)
                model_features.append(batch_features)

            model_features = np.array(model_features)
            model_features = np.vstack(model_features)
            model_features = np.squeeze(model_features)


            if l != -1:
                num_filters = model_features.shape[-1]
                model_features = model_features.reshape(num_patches,-1,num_filters)
                for cagg in convolutional_aggregations:
                    feature_path = GTEx_directory + '/data/features/{tissue}/{model_choice}_{ID}_{patch_size}_l{layer}_ca-{cagg}.hdf5' 
                    feature_path = feature_path.format(tissue=tissue,model_choice=model_choice,ID=ID,layer=l,patch_size=ps,cagg=cagg)
                    print ('Convolutional aggregation: {}'.format(cagg))
                    model_features1 = eval('np.{}(model_features, axis=1)'.format(cagg))

                    with h5py.File(feature_path,'w') as h:
                        h.create_dataset('features', data=model_features1)
            else:

                feature_path = GTEx_directory + '/data/features/{tissue}/{model_choice}_{ID}_{patch_size}_l{layer}.hdf5'
                feature_path = feature_path.format(tissue=tissue,model_choice=model_choice,ID=ID,patch_size=ps,layer=l)

                with h5py.File(feature_path,'w') as h:
                    h.create_dataset('features', data=model_features)

