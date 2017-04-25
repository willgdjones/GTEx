import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pdb

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


    tissue_expression_filepath = '/nfs/research2/stegle/stegle_secure/GTEx/download/49139/PhenoGenotypeFiles/RootStudyConsentSet_phs000424.GTEx.v6.p1.c1.GRU/ExpressionFiles/phe000006.v2.GTEx_RNAseq.expression-data-matrixfmt.c1/parse_data/44_tissues/GTEx_Data_20150112_RNAseq_RNASeQCv1.1.8_gene_rpkm_{}_normalised_without_inverse_gene_expression.txt'.format(tissue_filename)


    ID_representations = {}
    if not os.path.isfile('data/processed/association_data/expression/{}/{}/{}/X_y_{}'.format(model_name,tile_size, tissue, agg_method)):

        with open(tissue_expression_filepath, 'r') as f:
            expression_table = np.array([x.split('\t') for x in f.read().splitlines()])
            expression_matrix = expression_table[1:,1:].astype(np.float32)

        X = []
        Y = []
        for (j, ID) in enumerate(IDs):
            print (ID, len(IDs) - j)
            reps = pickle.load(open('data/processed/assembled_representations/{}/{}/{}/{}'.format(model_name, tile_size, tissue, ID), 'rb'))


            individual_tissue_expression_donor_IDs = [x.split('-')[1] for x in expression_table[0,:][1:]]
            # print (len(individual_tissue_expression_donor_IDs)) #278

            dID = str(ID).split('-')[1]
            # print (dID)

            try:
                ID_idx = individual_tissue_expression_donor_IDs.index(dID)
                X_row = expression_matrix[:,ID_idx]
                if agg_method == 'median':
                    Y_row = np.median(reps, axis=0)
                elif agg_method == 'mean':
                    Y_row = np.mean(reps, axis=0)
                X.append(X_row)
                Y.append(Y_row)
            except ValueError as e:
                print(e)
                continue
                    
        X = np.vstack(X)
        Y = np.vstack(Y)
        os.makedirs('data/processed/association_data/expression/{}/{}/{}'.format(model_name,tile_size,tissue), exist_ok = True)
        assert X.shape[0] == Y.shape[0]
        pickle.dump([X,Y], open('data/processed/association_data/expression/{}/{}/{}/X_y_{}'.format(model_name, tile_size, tissue, agg_method),'wb'))
    else:
        print ("X y matrix for {} {} {} {} already exists".format(model_name, tile_size, tissue, agg_method))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Program to assemble all covering representation batches into a single array')
    parser.add_argument('-t','--tissue', help='Tissue to assemble', required=True)
    parser.add_argument('-m','--model_name', help='Model that was used to generate representations', required=True)
    parser.add_argument('-a','--agg_method', help='Aggregration method to use, either mean or median.', default='mean')
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
    

