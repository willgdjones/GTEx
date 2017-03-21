import pickle
import matplotlib.pyplot as plt
import os
import numpy as np

import sys
sys.path = ['/nfs/gns/homes/willj/anaconda3/envs/GTEx/lib/python3.5/site-packages'] + sys.path

from keras.models import Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D, GlobalAveragePooling2D, Input, Dense
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
import argparse



tissue_types = ['Lung', 'Artery - Tibial', 'Heart - Left Ventricle', 'Breast - Mammary Tissue', 'Brain - Cerebellum', 'Pancreas', 'Testis', 'Liver', 'Ovary', 'Stomach']
genotypes_filepath = '/nfs/research2/stegle/stegle_secure/GTEx/download/49139/PhenoGenotypeFiles/RootStudyConsentSet_phs000424.GTEx.v6.p1.c1.GRU/GenotypeFiles/phg000520.v2.GTEx_MidPoint_Imputation.genotype-calls-vcf.c1/parse_data/GTEx_Analysis_20150112_OMNI_2.5M_5M_450Indiv_chr1to22_genot_imput_info04_maf01_HWEp1E6_ConstrVarIDs_all_chrom_filered_maf_subset_individuals_44_tissues.hdf5'
expression_filepath = '/nfs/research2/stegle/stegle_secure/GTEx/download/49139/PhenoGenotypeFiles/RootStudyConsentSet_phs000424.GTEx.v6.p1.c1.GRU/ExpressionFiles/phe000006.v2.GTEx_RNAseq.expression-data-matrixfmt.c1/parse_data/44_tissues/GTEx_Data_20150112_RNAseq_RNASeQCv1.1.8_gene_rpkm_*_normalised_without_inverse_gene_expression.txt'
phenotype_filepath = '/nfs/research2/stegle/stegle_secure/GTEx/download/49139/PhenoGenotypeFiles/RootStudyConsentSet_phs000424.GTEx.v6.p1.c1.GRU/PhenotypeFiles/phs000424.v6.pht002743.v6.p1.c1.GTEx_Sample_Attributes.GRU.txt.gz'

def build_empty_model():
    inception_model = InceptionV3(weights='imagenet', include_top=False)
    x = inception_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(10, activation='softmax')(x)

    model = Model(input=inception_model.input, output=predictions)
    return model

def main():
    model = build_empty_model()
    model.load_weights(model_path)

    final_layer_model = Model(model.input, model.layers[-2].output)


    for ID in os.listdir('data/processed/covering_patches/{}'.format(tissue)):
        os.makedirs('data/processed/assembled_representations/{}/{}'.format(model_name,tissue), exist_ok=True)
        if ID not in os.listdir('data/processed/assembled_representations/{}/{}/'.format(model_name,tissue)):
            for batch in os.listdir('data/processed/covering_patches/{}/{}'.format(tissue, ID)):
                print (batch)
                reps = []
                try:
                    x = np.array(pickle.load(open('data/processed/covering_patches/{}/{}/{}'.format(tissue,ID, batch), 'rb')))
                    x_reps = final_layer_model.predict(x)
                    reps.append(x_reps)
                    # pickle.dump(x, open('data/processed/representations/{}/{}/{}/{}'.format(model_name,tissue,ID, batch),'wb'))
                except ValueError:
                    continue
                reps = np.vstack(reps)
                pickle.dump(reps, open('data/processed/assembled_representations/{}/{}/{}'.format(model_name,tissue, ID),'wb')) 
            
                
        else:
            print ("Representations already exist")
            continue
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This model 1. Creates representations given a model and a tissue. 2. Assembled and saves these representations')
    parser.add_argument('-t','--tissue', help='The tissue', required=True)
    parser.add_argument('-m','--model_path', help='Path to the model used to generate the representations', required=True)
    args = vars(parser.parse_args())
    tissue = args['tissue'] 
    model_path = args['model_path']
    model_name = model_path.split('/')[-1]
    print (model_path, model_name)
    main()

