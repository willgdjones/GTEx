
import sys
import os
import numpy as np
import h5py
import pdb
from scipy.misc import imresize
import matplotlib.pyplot as plt

GTEx_directory = '/hps/nobackup/research/stegle/users/willj/GTEx'
sys.path = ['/nfs/gns/homes/willj/anaconda3/envs/GTEx/lib/python3.5/site-packages'] + sys.path
tissue_types = ['Lung', 'Artery - Tibial', 'Heart - Left Ventricle', 'Breast - Mammary Tissue', 'Brain - Cerebellum', 'Pancreas', 'Testis', 'Liver', 'Ovary', 'Stomach']

def get_donor_IDs(IDlist):
    return [str(x).split('-')[1] for x in IDlist]

genotypes_filepath = '/nfs/research2/stegle/stegle_secure/GTEx/download/49139/PhenoGenotypeFiles/RootStudyConsentSet_phs000424.GTEx.v6.p1.c1.GRU/GenotypeFiles/phg000520.v2.GTEx_MidPoint_Imputation.genotype-calls-vcf.c1/parse_data/GTEx_Analysis_20150112_OMNI_2.5M_5M_450Indiv_chr1to22_genot_imput_info04_maf01_HWEp1E6_ConstrVarIDs_all_chrom_filered_maf_subset_individuals_44_tissues.hdf5'
with h5py.File(genotypes_filepath, 'r') as f:
    genotype_IDs = f['genotype']['row_header']['sample_ID'].value
    genotype_donorIDs = get_donor_IDs(genotype_IDs)
    genotype_matrix = np.array(f['genotype']['matrix'].value)



g = h5py.File(GTEx_directory + '/data/hdf5/aggregated_features.hdf5','w')
for t in tissue_types:

    tissue_filepath = os.path.join(GTEx_directory,'data','raw',t)
    tissue_images = os.listdir(tissue_filepath)
    tissue_IDs = [x.split('.')[0] for x in tissue_images]

    tissue_expression_filepath = '/nfs/research2/stegle/stegle_secure/GTEx/download/49139/PhenoGenotypeFiles/RootStudyConsentSet_phs000424.GTEx.v6.p1.c1.GRU/ExpressionFiles/phe000006.v2.GTEx_RNAseq.expression-data-matrixfmt.c1/parse_data/44_tissues/GTEx_Data_20150112_RNAseq_RNASeQCv1.1.8_gene_rpkm_{}_normalised_without_inverse_gene_expression.txt'.format(tissue)
    if tissue == 'Artery - Tibial':
        tissue_filename = 'Artery_Tibial'
    elif tissue == 'Heart - Left Ventricle':
        tissue_filename = 'Heart_Left_Ventricle'
    elif tissue == 'Breast - Mammary Tissue':
        tissue_filename = 'Breast_Mammary_Tissue'
    elif tissue == 'Brain - Cerebellum':
        tissue_filename = 'Brain_Cerebellum'
    else:
        tissue_filename = tissue

    with open(tissue_expression_filepath, 'r') as f:
        expression_table = np.array([x.split('\t') for x in f.read().splitlines()])
        expression_matrix = expression_table[1:,1:].astype(np.float32)
        tissue_expression_donor_IDs = [x.split('-')[1] for x in expression_table[0,:][1:]]


    aggregations = ['mean','median','max']
    patch_sizes = [128,256,512,1024,2048,4096]
    model_choices = ['raw','retrained']

    for s in patch_sizes:
        for m in model_choice:
            for a in aggregations:
                ordered_features = []
                ordered_expression = []
                ordered_genotype = []
                donorIDs = []
                for (k,ID) in enumerate(tissue_IDs):

                    try:
                        donorID = str(ID).split('-')[1]
                        donorID_exp_idx = tissue_expression_donor_IDs.index(donorID)
                        donorID_gen_idx = genotype_donorIDs.index(donorID)
                        expression_row = expression_matrix[:,donorID_exp_idx]
                        genotype_row = genotype_matrix[:,donorID_gen_idx]
                        donorIDs.append(donorIDs)
                        print (ID, len(tissue_IDs) - k)
                    except ValueError, e
                        print (e)
                        continue


                    filename = '/data/features/{tissue}/{model_choice}_{ID}_{size}.hdf5'.format(tissue=t,model_choice=m,ID=ID,size=s)
                    f = h5py.File(os.path.join(filepath,filename),'r')
                    features = f['features'].value
                    f.close()

                    ordered_features.append(features)
                    ordered_expression.append(expression_row)
                    ordered_genotype.append(genotype_row)


                group = g.create_group('{tissue}/{model_choice}/{size}/{aggregation}'.format(tissue=tissue,model_choice=model_choice,size=size,aggregation=a))
                group.create_dataset('features', data=ordered_features)
                group.create_dataset('expression', data=ordered_expression)
                group.create_dataset('genotype', data=ordered_genotype)
                group.create_dataset('donorIDs', data=genotype_matrix)

g.close()
