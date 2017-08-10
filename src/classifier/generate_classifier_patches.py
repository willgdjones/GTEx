#Preprocessing
import sys
sys.path = ['/nfs/gns/homes/willj/anaconda3/envs/GTEx/lib/python3.5/site-packages'] + sys.path
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pickle
import gzip
import pandas as pd
import requests
import openslide
from openslide.deepzoom import DeepZoomGenerator
from openslide import open_slide
import math
import pdb
import time
import os
import argparse
import h5py
from scipy.misc.pilutil import imresize
import sys

def get_donor_IDs(IDlist):
    return [str(x).split('-')[1] for x in IDlist]

def sample_tiles_from_image(tile_size,tile_number,image_path):
    #Sample n tiles of size mxm from image
    sampled_tiles = []
    try:
        slide = open_slide(os.path.join(GTEx_directory,image_path))
        tiles = DeepZoomGenerator(slide, tile_size=tile_size, overlap=0, limit_bounds=False)
        tile_level = range(len(tiles.level_tiles))[tile_level_index]
        tile_dims = tiles.level_tiles[tile_level_index]
        count = 0

        t = time.time()
        # expect sampling rate to be at least 1 tile p/s. If time take is greater than this, move to next image.     
        while (count < tile_number and (time.time() - t < tile_number * 2)):
            #retreive tile
            tile = tiles.get_tile(tile_level, (np.random.randint(tile_dims[0]), np.random.randint(tile_dims[1])))
            image = np.array(tile.getdata(), dtype=np.float32).reshape(tile.size[0],tile.size[1],3)
            #calculate mean pixel intensity
            mean_pixel = np.mean(image.flatten())
            image = imresize(image,(299,299))
            if mean_pixel > 230:
                continue
            elif mean_pixel <= 230:
                sampled_tiles.append(image)
                count += 1

        if (time.time() - t > tile_number * 2):
            print("Timeout")
    except Exception as e:
        print("Error")

    return sampled_tiles

# Use 100 Random images that are already downloaded
random_tissue_ID_lookup = {}
for t1 in list(tissue_ID_lookup.keys()):
    available_image_IDs = [x.split('.')[0] for x in os.listdir(os.path.join(GTEx_directory,'data','raw', t1))]
    random_choice = np.random.choice(available_image_IDs,100)
    random_tissue_ID_lookup[t1] = random_choice

#ensure all images are downloaded
truth_array = []
for t2 in list(random_tissue_ID_lookup.keys()):
    for ID in random_tissue_ID_lookup[t2]:
        truth_array.append(os.path.isfile(os.path.join(GTEx_directory,'data','raw',t2,ID + '.svs')))

assert all(truth_array) == True


def main():
    tile_size = 128
    for (k,t3) in enumerate(list(random_tissue_ID_lookup.keys())):
        labels = []
        IDs = []
        data = []
        print (t3)
        if os.path.isfile('data/processed/patches/data_{}_{}_{}.py'.format(t3,tile_number,tile_level_index)):
            print ('file exists')
            continue

        for (i,ID) in enumerate(random_tissue_ID_lookup[t3]):
            image_path = os.path.join(GTEx_directory,'data','raw',t3,ID + '.svs')
            t = time.time()
            tiles = sample_tiles_from_image(tile_size,tile_number,image_path)
            data.extend(tiles)
            labels.extend([t3] * len(tiles))
            IDs.extend([ID] * len(tiles))
            print (ID,len(tiles),i)
        print ('{} done. Data length: {}, Labels length: {}'.format(t3, len(data), len(labels)))
        data = (np.array(data,dtype=np.float16) / 255)
        pickle.dump([data,labels, IDs], open('data/processed/patches/data_{}_{}_{}.py'.format(t3,tile_number,tile_level_index), 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate the image patches that are fed into the tissues classifier')
    parser.add_argument('-n','--tile_number', help='The number of patches sampled from each image', required=True)
    parser.add_argument('-l','--tile_level_index', help='Description for foo argument', required=True)
    args = vars(parser.parse_args())
    tile_number = int(args['tile_number'])
    tile_level_index = int(args['tile_level_index'])
    main()
