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
import os
GTEx_directory = '/hps/nobackup/research/stegle/users/willj/GTEx'
import mahotas
import sys
import argparse

def main():
    image_path = os.path.join(GTEx_directory,'data','raw',tissue,ID + '.svs')
    tile_size = 128
    tile_level_index = -1
    slide = open_slide(image_path)

    tiles = DeepZoomGenerator(slide, tile_size=tile_size, overlap=0, limit_bounds=False)
    tile_level = range(len(tiles.level_tiles))[tile_level_index]
    tile_dims = tiles.level_tiles[tile_level_index]

    tile_stack = []
    k = 0
    os.makedirs(os.path.join('data','processed','covering_patches',tissue,ID), exist_ok = True)
    if os.listdir(os.path.join('data','processed','covering_patches',tissue,ID)) == []: 
        for i in range(tile_dims[0]):
            for j in range(tile_dims[1]):
                tile = np.array(tiles.get_tile(tile_level, (i, j)))
                #If mean pixel > 0
                if np.mean(tile.flatten()) < 230:
                    tile_stack.append(tile)
                    if k % 2000 == 0 and k > 0:
                        pickle.dump(tile_stack,open(os.path.join('data','processed','covering_patches',tissue,ID,'{}_{}'.format(ID,k)), 'wb'))
                        tile_stack = []
                    print (k, i, j)                
                    k += 1
                else:
                    #Otherwise we are in whitespace
                    continue
    else:
        print ("Already generated data for {} {}".format(tissue, ID))

                    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate the covering patches for an image.')
    parser.add_argument('-i','--information', help='Takes the form "ID TISSUE". ', required=True)
    args = vars(parser.parse_args())
    information = args['information'].split(' ')
    ID = information[0]
    tissue = ' '.join(information[1:])
    print (ID, tissue)
    main()
