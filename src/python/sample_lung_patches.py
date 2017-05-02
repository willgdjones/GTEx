import os
import sys
sys.path = ['/nfs/gns/homes/willj/anaconda3/envs/GTEx/lib/python3.5/site-packages'] + sys.path
GTEx_directory = '/hps/nobackup/research/stegle/users/willj/GTEx'
import openslide
from openslide.deepzoom import DeepZoomGenerator
from openslide import open_slide
import numpy as np
import matplotlib.pyplot as plt
# from keras.layers import Input, GlobalAveragePooling2D, Dense
# from keras.models import Model
# from keras.applications.inception_v3 import InceptionV3
import h5py
import cv2
import mahotas
import ipdb
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('-f','--data_filename', help='Name of the filename to save to', required=True)
parser.add_argument('-i','--ID', help='GTEx ID', required=True)
args = vars(parser.parse_args())
data_filename = args['data_filename']
ID = args['ID']

lung_filepath = os.path.join(GTEx_directory,'data','raw','Lung')
lung_images = os.listdir(lung_filepath)
lung_IDs = [x.split('.')[0] for x in lung_images]

tissue = 'Lung'
tissue_expression_filepath = '/nfs/research2/stegle/stegle_secure/GTEx/download/49139/PhenoGenotypeFiles/RootStudyConsentSet_phs000424.GTEx.v6.p1.c1.GRU/ExpressionFiles/phe000006.v2.GTEx_RNAseq.expression-data-matrixfmt.c1/parse_data/44_tissues/GTEx_Data_20150112_RNAseq_RNASeQCv1.1.8_gene_rpkm_{}_normalised_without_inverse_gene_expression.txt'.format(tissue)
with open(tissue_expression_filepath, 'r') as f:
    expression_table = np.array([x.split('\t') for x in f.read().splitlines()])
    expression_matrix = expression_table[1:,1:].astype(np.float32)

individual_tissue_expression_donor_IDs = [x.split('-')[1] for x in expression_table[0,:][1:]]

def build_empty_model():
    inception_model = InceptionV3(weights='imagenet', include_top=False)
    x = inception_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(10, activation='softmax')(x)

    model = Model(input=inception_model.input, output=predictions)
    return model

def sample_patches(ID,patchsize):

    image_filepath = os.path.join(GTEx_directory,'data','raw','Lung', ID + '.svs')

    image_slide = open_slide(image_filepath)
    toplevel = image_slide.level_count - 1
    topdim = image_slide.level_dimensions[-1]
    topdownsample = image_slide.level_downsamples[-1]
    topdownsampleint = int(topdownsample)

    toplevelslide = image_slide.read_region((0,0), toplevel, topdim)
    toplevelslide = np.array(toplevelslide)
    toplevelslide = toplevelslide[:,:,0:3]
    slide = toplevelslide

    blurredslide = cv2.GaussianBlur(slide, (51,51),0)
    blurredslide = cv2.cvtColor(blurredslide, cv2.COLOR_BGR2GRAY)
    T_otsu = mahotas.otsu(blurredslide)

    mask = np.zeros_like(slide)
    mask = mask[:,:,0]
    mask[blurredslide < T_otsu] = 255


    downsampledpatchsize = patchsize / topdownsampleint
    xlimit = int(topdim[1] / downsampledpatchsize)
    ylimit = int(topdim[0] / downsampledpatchsize)


    # Find downsampled coords
    coords = []
    for i in range(xlimit):
        for j in range(ylimit):
            x = int(downsampledpatchsize/2 + i*downsampledpatchsize)
            y = int(downsampledpatchsize/2 + j*downsampledpatchsize)
            coords.append((x,y))

    # Find coords in downsampled mask
    mask_coords = []
    for c in coords:
        x = c[0]
        y = c[1]
        if mask[x,y] > 0:
            mask_coords.append(c)
            
    slidemarkings = slide.copy()
    for c in mask_coords:
        x = c[0]
        y = c[1]
        slidemarkings[x-2:x+2,y-2:y+2] = [0,0,255]
        


    # Convert downsampled masked coords to full masked coords
    full_mask_coords = []
    for c in mask_coords:
        x = c[0]
        y = c[1]
        full_x = int(topdownsample*x)
        full_y = int(topdownsample*y)
        full_mask_coords.append((full_x,full_y))

    print ('Number of tiles: {}'.format(len(full_mask_coords)))

    tiles = DeepZoomGenerator(image_slide, tile_size=patchsize, overlap=0, limit_bounds=False)

    covering_tiles = []
    for (i,full_coord) in enumerate(full_mask_coords):
        if i % 1000 == 0:
            print (i)
        x = int(full_coord[1] / patchsize)
        y = int(full_coord[0] / patchsize)
        tile = tiles.get_tile(tiles.level_count - 1,(x,y))
        tile = np.array(tile)
        covering_tiles.append(tile)

    covering_tiles = np.array(covering_tiles)
    return covering_tiles



# model = build_empty_model()
# model.load_weights(os.path.join('models','inception_50_-1.h5'))

# final_layer_model = Model(model.input, model.layers[-2].output)


# f = h5py.File(data_filename,'w')
# expression = f.create_group('lung/expression')
patch_sizes = [128, 256, 512, 1024, 2048]
# for patch_size in patch_sizes:
    # f.create_group('lung/patches/'+str(patch_size))
    # f.create_group('lung/features/'+str(patch_size))


# expression.create_dataset('transcript_names', data=expression_table[1:,0])

#Check that we have expression data
donorID = str(ID).split('-')[1]
ID_idx = individual_tissue_expression_donor_IDs.index(donorID)
expression_row = expression_matrix[:,ID_idx]
# expression.create_dataset(ID, data=expression_row)

for patch_size in patch_sizes:
    print (ID)

    print ('sampling patches for {}, {}'.format(ID, patch_size))

    covering_tiles = sample_patches(ID, patch_size)
    f = h5py.File('data/better_covering_patches/{}_{}.hdf5'.format(ID,patch_size), 'w')
    f.create_dataset('patches', data=covering_tiles)
    f.close()
    # patches.create_dataset(ID, data=covering_tiles)

    # print ('generating features for {}, {}'.format(ID, patch_size))
    # features = f['lung/features/'+str(patch_size)]
    # image_features = []
    # upper_limit = int(len(covering_tiles)/50)
    # for i in range(upper_limit):
        # if i % 100 == 0:
            # print (10*i)
        # batch_tiles = covering_tiles[i*10:(i+1)*10,:,:,:].astype(np.float16) / 255
        # batch_features = final_layer_model.predict(batch_tiles)
        # assert batch_features.shape[1] == 1024
        # image_features.append(batch_features)
    # image_features = np.array(image_features)
    # features.create_dataset(ID, data=image_features)

       

            





