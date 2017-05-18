import os
import sys
import openslide
from openslide.deepzoom import DeepZoomGenerator
from openslide import open_slide
import numpy as np
import matplotlib.pyplot as plt
import h5py
import cv2
import mahotas
import pdb
import argparse

sys.path = ['/nfs/gns/homes/willj/anaconda3/envs/GTEx/lib/python3.5/site-packages'] + sys.path
GTEx_directory = '/hps/nobackup/research/stegle/users/willj/GTEx'

patch_sizes = [128, 256, 512, 1024, 2048, 4096]


parser = argparse.ArgumentParser(description='')
parser.add_argument('-p','--pair', help='GTEx ID, Tissue pair', required=True)
args = vars(parser.parse_args())
pair = args['pair'].split(' ')
tissue = ' '.join(pair[1:])
ID = pair[0]

tissue_filepath = os.path.join(GTEx_directory,'data','raw',tissue)
tissue_images = os.listdir(tissue_filepath)



def sample_patches(ID,patchsize):

    image_filepath = os.path.join(GTEx_directory,'data','raw',tissue, ID + '.svs')

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


for patch_size in patch_sizes:
    filepath = os.path.join('data/patches',tissue)
    filename = '{ID}_{patch_size}.hdf5'.format(ID=ID,patch_size=patch_size)
    os.makedirs(filepath, exist_ok=True)


    if not os.path.isfile(os.path.join(filepath,filename)):
        print ('sampling patches for {}, {}'.format(ID, patch_size))
        covering_tiles = sample_patches(ID, patch_size)
        f = h5py.File(os.path.join(filepath,filename), 'w')
        f.create_dataset('patches', data=covering_tiles)
        f.close()
    else:
        print ('File exists')
