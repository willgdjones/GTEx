import gzip
import pandas as pd
import numpy as np
import h5py
import os
import requests
import argparse
import pdb

genotypes_filepath = '/nfs/research2/stegle/stegle_secure/GTEx/download/49139/PhenoGenotypeFiles/RootStudyConsentSet_phs000424.GTEx.v6.p1.c1.GRU/GenotypeFiles/phg000520.v2.GTEx_MidPoint_Imputation.genotype-calls-vcf.c1/parse_data/GTEx_Analysis_20150112_OMNI_2.5M_5M_450Indiv_chr1to22_genot_imput_info04_maf01_HWEp1E6_ConstrVarIDs_all_chrom_filered_maf_subset_individuals_44_tissues.hdf5'
expression_filepath = '/nfs/research2/stegle/stegle_secure/GTEx/download/49139/PhenoGenotypeFiles/RootStudyConsentSet_phs000424.GTEx.v6.p1.c1.GRU/ExpressionFiles/phe000006.v2.GTEx_RNAseq.expression-data-matrixfmt.c1/parse_data/44_tissues/GTEx_Data_20150112_RNAseq_RNASeQCv1.1.8_gene_rpkm_*_normalised_without_inverse_gene_expression.txt'
phenotype_filepath = '/nfs/research2/stegle/stegle_secure/GTEx/download/49139/PhenoGenotypeFiles/RootStudyConsentSet_phs000424.GTEx.v6.p1.c1.GRU/PhenotypeFiles/phs000424.v6.pht002743.v6.p1.c1.GTEx_Sample_Attributes.GRU.txt.gz'
GTEx_directory = '/hps/nobackup/research/stegle/users/willj/GTEx'

def get_donor_IDs(IDlist):
    return [str(x).split('-')[1] for x in IDlist]

#Phenotype
with gzip.open(phenotype_filepath, 'rb') as f:
    phenotype_df = pd.DataFrame([str(x, 'utf-8').split('\t') for x in f.read().splitlines() if not str(x, 'utf-8').startswith('#')][1:])
    phenotype_df.columns = phenotype_df.iloc[0]
    phenotype_df = phenotype_df[1:]
    phenotype_IDs = ['-'.join(x.split('-')[0:3]) for x in list(phenotype_df['SAMPID']) if x.startswith('GTEX')]
    donor_phenotype_IDs = get_donor_IDs(phenotype_IDs)
    unique_donor_phenotype_IDs = np.unique(donor_phenotype_IDs)
    # 11983 phenotype IDs. 

#Images
with open('data/aperio-downloader/gtex-sampid.txt','r') as f:
    image_IDs = f.read().splitlines()
    donor_image_IDs = get_donor_IDs(image_IDs)
    unique_donor_image_IDs = np.unique(donor_image_IDs)
    # 22642 images
    # 868 unique image donors
    # 571 unique phenotype donors


#Genotype
with h5py.File(genotypes_filepath,'r') as f:
    genotype_IDs = f['genotype']['row_header']['sample_ID']
    donor_genotype_IDs = get_donor_IDs(genotype_IDs)
    unique_donor_genotype_IDs = np.unique(donor_genotype_IDs)
    # 449 genotypes
    # 449 unique genotype donors


# Define tissue types of interest
tissue_types = ['Lung', 'Artery - Tibial', 'Heart - Left Ventricle', 'Breast - Mammary Tissue', 'Brain - Cerebellum', 'Pancreas', 'Testis', 'Liver', 'Ovary', 'Stomach']

# Lookup tissues IDs from phenotype_df
tissue_IDs_set = [x for x in [list(phenotype_df[phenotype_df['SMTSD'] == tissue]['SAMPID']) for tissue in tissue_types]]
tissue_image_IDs = [list(set(['-'.join(x.split('-')[:3]) for x in tissue_IDs]).intersection(set(image_IDs))) for tissue_IDs in tissue_IDs_set]
tissue_ID_lookup = dict((tissue_types[i], tissue_image_IDs[i]) for i in range(len(tissue_image_IDs)))

#Expression / Genotype numbers
# Lung 295 277
# Artery - Tibial 302 280
# Heart - Left Ventricle 212 189
# Breast - Mammary Tissue 186 182
# Brain - Cerebellum 109 102
# Pancreas 168 148
# Testis 177 155
# Liver 112 97
# Ovary 95 83
# Stomach 176 168
def genotype_expression_numbers():
    for tissue in tissue_types:
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

        tissue_expression_filepath = '/nfs/research2/stegle/stegle_secure/GTEx/download/49139/PhenoGenotypeFiles/RootStudyConsentSet_phs000424.GTEx.v6.p1.c1.GRU/ExpressionFiles/phe000006.v2.GTEx_RNAseq.expression-data-matrixfmt.c1/parse_data/44_tissues/GTEx_Data_20150112_RNAseq_RNASeQCv1.1.8_gene_rpkm_{}_normalised_without_inverse_gene_expression.txt'.format(tissue_filename)
        with open(tissue_expression_filepath, 'r') as f:
            expression_table = np.array([x.split('\t') for x in f.read().splitlines()])
            expression_matrix = expression_table[1:,1:].astype(np.float32)
            individual_tissue_expression_donor_IDs = [x.split('-')[1] for x in expression_table[0,:][1:]]

        number_genotypes = len(list(set(get_donor_IDs(tissue_ID_lookup[tissue])).intersection(set(unique_donor_genotype_IDs))))
        number_expression = len(list(set(get_donor_IDs(tissue_ID_lookup[tissue])).intersection(set(individual_tissue_expression_donor_IDs))))
        print (tissue, number_genotypes, number_expression)


def test_image_genotype_intersection():
    assert len(set(unique_donor_image_IDs).intersection(unique_donor_genotype_IDs)) == 447

def test_image_phenotype_intersection():
    assert len(set(unique_donor_image_IDs).intersection(unique_donor_phenotype_IDs)) == 567

def test_genotype_phenotype_intersection():
    assert len(set(unique_donor_genotype_IDs).intersection(unique_donor_phenotype_IDs)) == 449

def test_genotype_phenotype_image_intersection():
    assert len(set(unique_donor_genotype_IDs).intersection(unique_donor_phenotype_IDs).intersection(set(unique_donor_image_IDs))) == 447
