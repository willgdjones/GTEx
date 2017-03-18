
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

def download_tissues(random_tissue_ID_lookup):
    for t4 in list(random_tissue_ID_lookup.keys()):
        random_image_IDs = random_tissue_ID_lookup[t4]
        try:
            os.mkdir(os.path.join('data','raw',t4))
        except:
            print('Directory exists')
        print('Downloading tissue: {}'.format(t4))
        download(random_image_IDs, os.path.join('data','raw',t4))



def openList(samples_filename):
    with open(samples_filename) as f:
        samples_list = f.read().splitlines()
    return(samples_list)

def download(samples_list, directory):
    print("Downloading images...")
    session = requests.session()

    # Do download
    for i in range(0, len(samples_list)):
        
        output_filename = os.path.join(directory,samples_list[i]+".svs")
        if not os.path.isfile(output_filename):
            print("Downloading "+str(i+1)+"/"+str(len(samples_list))+": "+samples_list[i])
            with open(output_filename, 'wb') as outfile:
                URL = "https://brd.nci.nih.gov/brd/imagedownload/" + str(samples_list[i])
                response = session.get(URL)
                if not response.ok:
                    print("Could not download")
                    raise Exception
                else:
                    outfile.write(response.content)
        else:
            print(str(i+1)+"/"+str(len(samples_list))+": "+samples_list[i] +" already Downloaded")
    return 


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
    #     f,a = plt.subplots(4,4,figsize=(10,10))
        count = 0
        
        t = time.time()
        # expect sampling rate to be at least 1 tile p/s. If time take is greater than this, move to next image.
    #         
        while (count < tile_number and (time.time() - t < tile_number * 2)):
    #             print (time.time() - t)
            #retreive tile

            tile = tiles.get_tile(tile_level, (np.random.randint(tile_dims[0]), np.random.randint(tile_dims[1])))
            image = 255 - np.array(tile.getdata(), dtype=np.float32).reshape(tile.size[0],tile.size[1],3)
            #calculate mean pixel intensity
            mean_pixel = np.mean(image.flatten())
            image = imresize(image,(299,299))
            if mean_pixel < 20:
                continue
            elif mean_pixel >= 20:
    #             a.flatten()[count].axis('off')
    #             a.flatten()[count].set_title(mean_pixel, size=5)
    #             a.flatten()[count].imshow(image)
                sampled_tiles.append(image)
                count += 1
            
        if (time.time() - t > tile_number * 2):
            print("Timeout")
    except Exception as e:
        # print (sys.exc_info())
        print("Error")
        
    return sampled_tiles
# Directory/Filepath variables
genotypes_filepath = '/nfs/research2/stegle/stegle_secure/GTEx/download/49139/PhenoGenotypeFiles/RootStudyConsentSet_phs000424.GTEx.v6.p1.c1.GRU/GenotypeFiles/phg000520.v2.GTEx_MidPoint_Imputation.genotype-calls-vcf.c1/parse_data/GTEx_Analysis_20150112_OMNI_2.5M_5M_450Indiv_chr1to22_genot_imput_info04_maf01_HWEp1E6_ConstrVarIDs_all_chrom_filered_maf_subset_individuals_44_tissues.hdf5'
expression_filepath = '/nfs/research2/stegle/stegle_secure/GTEx/download/49139/PhenoGenotypeFiles/RootStudyConsentSet_phs000424.GTEx.v6.p1.c1.GRU/ExpressionFiles/phe000006.v2.GTEx_RNAseq.expression-data-matrixfmt.c1/parse_data/44_tissues/GTEx_Data_20150112_RNAseq_RNASeQCv1.1.8_gene_rpkm_*_normalised_without_inverse_gene_expression.txt'
phenotype_filepath = '/nfs/research2/stegle/stegle_secure/GTEx/download/49139/PhenoGenotypeFiles/RootStudyConsentSet_phs000424.GTEx.v6.p1.c1.GRU/PhenotypeFiles/phs000424.v6.pht002743.v6.p1.c1.GTEx_Sample_Attributes.GRU.txt.gz'
GTEx_directory = '/hps/nobackup/research/stegle/users/willj/GTEx'
#Preprocessing

#Phenotype
with gzip.open(phenotype_filepath, 'rb') as f:
    phenotype_df = pd.DataFrame([str(x, 'utf-8').split('\t') for x in f.read().splitlines() if not str(x, 'utf-8').startswith('#')][1:])
    phenotype_df.columns = phenotype_df.iloc[0]
    phenotype_df = phenotype_df[1:]
    phenotype_IDs = ['-'.join(x.split('-')[0:3]) for x in list(phenotype_df['SAMPID']) if x.startswith('GTEX')]
    donor_phenotype_IDs = get_donor_IDs(phenotype_IDs)
    unique_donor_phenotype_IDs = np.unique(donor_phenotype_IDs)
    # 11983 phenotype IDs. 
    # 571 unique phenotype donors
    
#Genotype
with h5py.File(genotypes_filepath,'r') as f:
    genotype_IDs = f['genotype']['row_header']['sample_ID']
    donor_genotype_IDs = get_donor_IDs(genotype_IDs)
    unique_donor_genotype_IDs = np.unique(donor_genotype_IDs)
    # 449 genotypes
    # 449 unique genotype donors

#Images
with open('data/aperio-downloader/gtex-sampid.txt','r') as f:
    image_IDs = f.read().splitlines()
    donor_image_IDs = get_donor_IDs(image_IDs)
    unique_donor_image_IDs = np.unique(donor_image_IDs)
    # 22642 images
    # 868 unique image donors

tissue_sample_counts = list(zip(phenotype_df['SMTSD'].value_counts().index, phenotype_df['SMTSD'].value_counts().tolist()))

# unique image donors U unique genotype donors = 447 
len(set(unique_donor_image_IDs).intersection(unique_donor_genotype_IDs))
# unique image donors U unique phenotype donors = 567
len(set(unique_donor_image_IDs).intersection(unique_donor_phenotype_IDs))
# unique genotype donors U unique phenotype donors = 449
len(set(unique_donor_genotype_IDs).intersection(unique_donor_phenotype_IDs))
# # unique genotype donors U unique phenotype donors U unique image donors  = 447
len(set(unique_donor_genotype_IDs).intersection(unique_donor_phenotype_IDs).intersection(set(unique_donor_image_IDs)))

# Define tissue types of interest
tissue_types = ['Lung', 'Artery - Tibial', 'Heart - Left Ventricle', 'Breast - Mammary Tissue', 'Brain - Cerebellum', 'Pancreas', 'Testis', 'Liver', 'Ovary', 'Stomach']

# Lookup tissues IDs from phenotype_df
tissue_IDs_set = [x for x in [list(phenotype_df[phenotype_df['SMTSD'] == tissue]['SAMPID']) for tissue in tissue_types]]
tissue_image_IDs = [list(set(['-'.join(x.split('-')[:3]) for x in tissue_IDs]).intersection(set(image_IDs))) for tissue_IDs in tissue_IDs_set]
tissue_ID_lookup = dict((tissue_types[i], tissue_image_IDs[i]) for i in range(len(tissue_image_IDs)))

#Total number of GTEx IDs for the tissues

# [('Lung', 497),
#  ('Artery - Tibial', 438),
#  ('Heart - Left Ventricle', 336),
#  ('Breast - Mammary Tissue', 222),
#  ('Brain - Cerebellum', 163),
#  ('Pancreas', 204),
#  ('Testis', 209),
#  ('Liver', 143),
#  ('Ovary', 112),
#  ('Stomach', 211)]
tissue_ID_counts = [(t, len(x)) for t,x in tissue_ID_lookup.items()]
np.random.seed(42)
lung_tissue_ID_lookup = {'Lung': tissue_ID_lookup['Lung']}
download_tissues(lung_tissue_ID_lookup)

def main():
    tile_size = 128
    for (k,t3) in enumerate(list(lung_tissue_ID_lookup.keys())):
        labels = []
        IDs = []
        data = []
        print (t3)
        if os.path.isfile('data/processed/patches/lung_data_{}_{}_{}.py'.format(t3,tile_number,tile_level_index)):
            print ('file exists')
            continue
            
        for (i,ID) in enumerate(lung_tissue_ID_lookup[t3]):
            image_path = os.path.join(GTEx_directory,'data','raw',t3,ID + '.svs')
            t = time.time()
            tiles = sample_tiles_from_image(tile_size,tile_number,image_path)
            data.extend(tiles)
            labels.extend([t3] * len(tiles))
            IDs.extend([ID] * len(tiles))
            print (ID,len(tiles),i)
        print ('{} done. Data length: {}, Labels length: {}'.format(t3, len(data), len(labels)))
        data = (np.array(data,dtype=np.float16) / 255)
        pickle.dump([data,labels, IDs], open('data/processed/patches/lung_data_{}_{}_{}.py'.format(t3,tile_number,tile_level_index), 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate patches from available lung image IDs')
    parser.add_argument('-n','--tile_number', help='Description for foo argument', required=True)
    parser.add_argument('-l','--tile_level_index', help='Description for foo argument', required=True)
    args = vars(parser.parse_args())
    tile_number = int(args['tile_number'])
    tile_level_index = int(args['tile_level_index'])
    main()
