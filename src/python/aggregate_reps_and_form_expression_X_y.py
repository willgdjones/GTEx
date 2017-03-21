import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pdb

def get_donor_IDs(IDlist):
    return [str(x).split('-')[1] for x in IDlist]

def main():
    IDs = os.listdir('data/processed/assembled_representations/{}/{}'.format(model_name,tissue))
    tissue_expression_filepath = '/nfs/research2/stegle/stegle_secure/GTEx/download/49139/PhenoGenotypeFiles/RootStudyConsentSet_phs000424.GTEx.v6.p1.c1.GRU/ExpressionFiles/phe000006.v2.GTEx_RNAseq.expression-data-matrixfmt.c1/parse_data/44_tissues/GTEx_Data_20150112_RNAseq_RNASeQCv1.1.8_gene_rpkm_{}_normalised_without_inverse_gene_expression.txt'.format(tissue)

    ID_representations = {}
    if not os.path.isfile('data/processed/associations/expression/{}/{}/X_y_{}'.format(model_name, tissue, agg_method)):

        for (j, ID) in enumerate(IDs):
            print (ID, len(IDs) - j)
            reps = pickle.load(open('data/processed/assembled_representations/{}/{}/{}'.format(model_name,tissue, ID), 'rb'))
            ID_representations[ID] = reps

        with open(tissue_expression_filepath, 'r') as f:
            expression_table = np.array([x.split('\t') for x in f.read().splitlines()])
            expression_matrix = expression_table[1:,1:].astype(np.float32)

        individual_tissue_expression_donor_IDs = [x.split('-')[1] for x in expression_table[0,:][1:]]
        print (len(individual_tissue_expression_donor_IDs)) #278

        X = []
        Y = []
        for (gID, reps) in ID_representations.items():
            ID = str(gID).split('-')[1]
            print (ID)
            try:
                ID_idx = individual_tissue_expression_donor_IDs.index(ID)
                X_row = expression_matrix[:,ID_idx]
                if agg_method == 'median':
                    Y_row = np.median(ID_representations[gID], axis=0)
                elif agg_method == 'mean':
                    Y_row = np.mean(ID_representations[gID], axis=0)
                X.append(X_row)
                Y.append(Y_row)
            except ValueError as e:
                print(e)
                continue
                
        X = np.vstack(X)
        Y = np.vstack(Y)
        os.makedirs('data/processed/associations/expression/{}/{}'.format(model_name,tissue))
        pickle.dump([X,Y], open('data/processed/associations/expression/{}/{}/X_y_{}'.format(model_name, tissue, agg_method),'wb'))
    else:
        print ("X y matrix for {} {} {} already exists".format(model_name, tissue, agg_method))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Program to assemble all covering representation batches into a single array')
    parser.add_argument('-t','--tissue', help='Tissue to assemble', required=True)
    parser.add_argument('-m','--model_name', help='Model that was used to generate representations', required=True)
    parser.add_argument('-a','--agg_method', help='Aggregration method to use, either mean or median.', required=True)
    args = vars(parser.parse_args())
    tissue = args['tissue']
    model_name = args['model_name']
    agg_method = args['agg_method']
    assert agg_method == 'median' or agg_method == 'mean', "Aggregation method needs to be either the mean or median"
    print (tissue, model_name)
    main()
    

