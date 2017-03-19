import gzip
import pandas as pd
import numpy as np
import h5py
import os
import requests
import argparse

def get_donor_IDs(IDlist):
    return [str(x).split('-')[1] for x in IDlist]

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
    

#Images
with open('data/aperio-downloader/gtex-sampid.txt','r') as f:
    image_IDs = f.read().splitlines()
    donor_image_IDs = get_donor_IDs(image_IDs)
    unique_donor_image_IDs = np.unique(donor_image_IDs)
    # 22642 images
    # 868 unique image donors


# Define tissue types of interest
tissue_types = ['Lung', 'Artery - Tibial', 'Heart - Left Ventricle', 'Breast - Mammary Tissue', 'Brain - Cerebellum', 'Pancreas', 'Testis', 'Liver', 'Ovary', 'Stomach']

# Lookup tissues IDs from phenotype_df
tissue_IDs_set = [x for x in [list(phenotype_df[phenotype_df['SMTSD'] == tissue]['SAMPID']) for tissue in tissue_types]]
tissue_image_IDs = [list(set(['-'.join(x.split('-')[:3]) for x in tissue_IDs]).intersection(set(image_IDs))) for tissue_IDs in tissue_IDs_set]
tissue_ID_lookup = dict((tissue_types[i], tissue_image_IDs[i]) for i in range(len(tissue_image_IDs)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Program to download all image IDs for a tissue. Downloads to data/raw.')
    parser.add_argument('-t','--tissue', help='Tissue to download', required=True)
    args = vars(parser.parse_args())
    tissue = args['tissue']
    single_tissue_lookup = {tissue: tissue_ID_lookup[tissue]}
    download_tissues(single_tissue_lookup)
