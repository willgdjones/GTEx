import sys
import os
import numpy as np
import h5py
import pdb
from scipy.misc import imresize
import matplotlib.pyplot as plt

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
    inception_model = InceptionV3(weights='imagenet', include_top=False)
    x = inception_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(10, activation='softmax')(x)

    model = Model(input=inception_model.input, output=predictions)
    return model

len(os.listdir(os.path.join(GTEx_directory, 'data/better_covering_patches')))

lung_filepath = os.path.join(GTEx_directory,'data','raw','Lung')
lung_images = os.listdir(lung_filepath)

lung_IDs = [x.split('.')[0] for x in lung_images]

# tissue = 'Lung'
# tissue_expression_filepath = '/nfs/research2/stegle/stegle_secure/GTEx/download/49139/PhenoGenotypeFiles/RootStudyConsentSet_phs000424.GTEx.v6.p1.c1.GRU/ExpressionFiles/phe000006.v2.GTEx_RNAseq.expression-data-matrixfmt.c1/parse_data/44_tissues/GTEx_Data_20150112_RNAseq_RNASeQCv1.1.8_gene_rpkm_{}_normalised_without_inverse_gene_expression.txt'.format(tissue)
# with open(tissue_expression_filepath, 'r') as f:
    # expression_table = np.array([x.split('\t') for x in f.read().splitlines()])
    # expression_matrix = expression_table[1:,1:].astype(np.float32)

# individual_tissue_expression_donor_IDs = [x.split('-')[1] for x in expression_table[0,:][1:]]


model = build_empty_model()
model.load_weights(os.path.join(GTEx_directory, 'models','inception_50_-1.h5'))

final_layer_model = Model(model.input, model.layers[-2].output)



# donorID = str(ID).split('-')[1]
# ID_idx = individual_tissue_expression_donor_IDs.index(donorID)
# expression_row = expression_matrix[:,ID_idx]
# expression.create_dataset(ID, data=expression_row)
print (ID)

for ps in patch_sizes:
    # size = features[str(ps)]
    patch_path = GTEx_directory + '/data/better_covering_patches/{}_{}.hdf5'.format(ID,ps)
    g = h5py.File(patch_path, 'r')
    patches = g['patches']
#             g.close()
    patches = np.array([imresize(x, (299,299)) for x in patches])
    print (len(patches))
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

    assert np.array(image_features).shape[1] == 1024
    feature_path = GTEx_directory + '/data/better_image_features/{}_{}.hdf5'.format(ID,ps)
    with h5py.File(feature_path,'w') as h:
        h.create_dataset(ID, data=image_features)
