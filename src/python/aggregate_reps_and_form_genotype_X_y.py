import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pdb
import h5py

def get_donor_IDs(IDlist):
    return [str(x).split('-')[1] for x in IDlist]

def main():
    IDs = os.listdir('data/processed/assembled_representations/{}/{}/{}'.format(model_name,tile_size,tissue))
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


    genotypes_filepath = '/nfs/research2/stegle/stegle_secure/GTEx/download/49139/PhenoGenotypeFiles/RootStudyConsentSet_phs000424.GTEx.v6.p1.c1.GRU/GenotypeFiles/phg000520.v2.GTEx_MidPoint_Imputation.genotype-calls-vcf.c1/parse_data/GTEx_Analysis_20150112_OMNI_2.5M_5M_450Indiv_chr1to22_genot_imput_info04_maf01_HWEp1E6_ConstrVarIDs_all_chrom_filered_maf_subset_individuals_44_tissues.hdf5'


    ID_representations = {}
    if not os.path.isfile('data/processed/association_data/genotype/{}/{}/{}/X_y_{}'.format(model_name,tile_size, tissue, agg_method)):

        with h5py.File(genotypes_filepath, 'r') as f:
            genotype_IDs = f['genotype']['row_header']['sample_ID']
            donor_genotype_IDs = get_donor_IDs(genotype_IDs)
            unique_donor_genotype_IDs = np.unique(donor_genotype_IDs)
            genotype_matrix = np.array(f['genotype']['matrix'])
            genotype_matrix[genotype_matrix == 255] = 0

        X = []
        Y = []
        for (j, ID) in enumerate(IDs):
            print (ID, len(IDs) - j)
            reps = pickle.load(open('data/processed/assembled_representations/{}/{}/{}/{}'.format(model_name, tile_size, tissue, ID), 'rb'))
            ID_representations[ID] = reps

            dID = str(ID).split('-')[1]
            print (dID)
            try:
                ID_idx = donor_genotype_IDs.index(dID)
                X_row = genotype_matrix[ID_idx,:]
                try:
                    if agg_method == 'median':
                        Y_row = np.median(ID_representations[ID], axis=0)
                    elif agg_method == 'mean':
                        Y_row = np.mean(ID_representations[ID], axis=0)
                except IndexError as e:
                    pdb.set_trace()
                    print (e)
                    
                X.append(X_row)
                Y.append(Y_row)
                continue

            except ValueError as e:
                print (e)
                
        X = np.vstack(X)
        Y = np.vstack(Y)
        os.makedirs('data/processed/association_data/genotype/{}/{}/{}'.format(model_name,tile_size,tissue), exist_ok = True)
        assert X.shape[0] == Y.shape[0]
        pickle.dump([X,Y], open('data/processed/association_data/genotype/{}/{}/{}/X_y_{}'.format(model_name, tile_size, tissue, agg_method),'wb'))
    else:
        print ("X y matrix for {} {} {} {} already exists".format(model_name, tile_size, tissue, agg_method))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Program to assemble all covering representation batches into a single array')
    parser.add_argument('-t','--tissue', help='Tissue to assemble', required=True)
    parser.add_argument('-m','--model_name', help='Model that was used to generate representations', required=True)
    parser.add_argument('-a','--agg_method', help='Aggregration method to use, either mean or median.', required=True)
    parser.add_argument('-s','--tile_size', help='The tile of the patches', default='small')
    args = vars(parser.parse_args())
    tissue = args['tissue']
    model_name = args['model_name']
    agg_method = args['agg_method']
    assert agg_method == 'median' or agg_method == 'mean', "Aggregation method needs to be either the mean or median"
    tile_size = args['tile_size']
    assert tile_size == 'small' or tile_size == 'medium' or tile_size == 'large'
    if tile_size == 'small':
        tile_level_index = -1
    elif tile_size == 'medium':
        tile_level_index = -2
    elif tile_size == 'large':
        tile_level_index = -3
    else:
        raise Exception 
    print (tissue, tile_size, model_name)
    main()
    

