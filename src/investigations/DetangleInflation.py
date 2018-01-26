import os
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import h5py
import argparse
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from matplotlib.colors import Normalize
sys.path.insert(0, os.getcwd())
from src.utils.helpers import *


GTEx_directory = '/hps/nobackup/research/stegle/users/willj/GTEx'

parser = argparse.ArgumentParser(description='Collection of experiments. Runs on the cluster.')
parser.add_argument('-g', '--group', help='Experiment group', required=True)
parser.add_argument('-n', '--name', help='Experiment name', required=True)
args = vars(parser.parse_args())
group = args['group']
name = args['name']

import pickle
from limix.plot import qqplot
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import seaborn
GTEx_directory = '/hps/nobackup/research/stegle/users/willj/GTEx'
corr_dir = GTEx_directory + '/data/correction/'
sys.path.insert(0, GTEx_directory)
from src.utils.helpers import *
import pandas
import glob
from scipy.stats import norm
import h5py
from tqdm import tqdm
import scipy



# extract donors that have images
filelist = glob.glob(GTEx_directory + '/data/features/Lung/retrained*_256_l-1.hdf5')
donorlist = [x.split('/')[-1] for x in filelist]
donorlist = [x.split('_')[1] for x in donorlist]
donorlist = ['-'.join(x.split('-')[0:2]) for x in donorlist]

# extract donors with sample attributes
sample_attributes = pandas.read_table(GTEx_directory + '/data/GTEx_download/GTEx_v7_Annotations_SampleAttributesDS.txt')

sample_prep_covariates = sample_attributes[sample_attributes['SMTSD'] == 'Lung'][['SMRIN', 'SMTSISCH', 'SMATSSCR', 'SMNTRNRT', 'SMEXNCRT', 'SAMPID']].dropna()
# Take log of Ischemic time
sample_prep_covariates['SMTSISCH'] = np.log(sample_prep_covariates['SMTSISCH'])
sample_prep_covariates['SAMPID'] = ['-'.join(x.split('-')[0:2]) for x in list(sample_prep_covariates['SAMPID'])]
# Drop duplicate samples and take the one that has the highest RIN number
sample_prep_covariates = sample_prep_covariates.sort_values('SMRIN').drop_duplicates('SAMPID', keep='last')
sample_prep_covariates.index = sample_prep_covariates['SAMPID']
donors_with_sample_attributes = sample_prep_covariates['SAMPID']
sample_prep_covariates = sample_prep_covariates.drop('SAMPID', axis=1)
donors_with_sample_attributes = sample_prep_covariates.index.tolist()

# extract donors with expression
expression_table = pandas.read_table(GTEx_directory + '/data/GTEx_download/GTEx_Analysis_v7_eQTL_expression_matrices/Lung.v7.normalized_expression.bed',sep='\t')
donors_with_expression = list(expression_table.columns[4:])

# find intersection: 291 samples
donor_intersection = list(set(donors_with_sample_attributes).intersection(set(donors_with_expression)).intersection(set(donorlist)))

# build confounding and expression matrix for these donors
covariates = sample_prep_covariates.loc[donor_intersection]
expression = expression_table[donor_intersection]
expression = expression.T

# Read in confounding matrix table used for the GTEx eQTL analysis. Includes genotype PCs and PEER factors
PEER_table = pandas.read_table(GTEx_directory + '/data/GTEx_download/GTEx_Analysis_v7_eQTL_covariates/Lung.v7.covariates.txt',sep='\t')
PEER_idx = np.array([x.startswith('InferredCov') for x in list(PEER_table['ID'])])
# Extract the PEER factors for the donor intersection
PEER_factors = PEER_table[donor_intersection]
PEER_factors = PEER_factors.iloc[PEER_idx,:]
PEER_factors = PEER_factors.T

def create_ID(parameters):
    EXPg = int(parameters['expression']['gaussianize'])
    EXPc = parameters['expression']['correct']

    ACTg = int(parameters['activations']['gaussianize'])
    ACTc = parameters['activations']['correct']

    ID = 'EXPg{}c{}_ACTg{}c{}'.format(EXPg, EXPc, ACTg, ACTc)
    return ID

def estimate_lambda(pv):
    """estimate lambda form a set of PV"""
    LOD2 = sp.median(st.chi2.isf(pv, 1))
    null_median = st.chi2.median(1)
    L = (LOD2 / null_median)
    return L

def quantile_normalize_to_gaussian(x):
    """ """
    n = len(x)
    rank_x = scipy.stats.rankdata(x)
    unif = (rank_x + 1) / (n + 2)
    g_x = norm.ppf(unif)
    return g_x

def regress_out_k_covariates(activations, k):
    print ('Regressing out {} covariates from activations'.format(k))
    order_results = pickle.load(open(GTEx_directory + '/results/FeatureSelection/tf_feature_selection_image_features.pickle','rb'))
    ordered_cov = [x[1] for x in order_results]
    technical_confounders = ['SMRIN', 'SMTSISCH', 'SMATSSCR', 'SMNTRNRT', 'SMEXNCRT']
    choice_of_confounders = covariates[technical_confounders[0:k]].as_matrix()
    lr = LinearRegression()
    lr.fit(choice_of_confounders, activations)
    res_activations = activations - lr.predict(choice_of_confounders)
    return res_activations


def regress_out_k_peer_factors(expression, k):
    print ('Regressing out {} PEER factors from expression'.format(k))
    lr = LinearRegression()
    lr.fit(PEER_factors.iloc[:,:k].as_matrix(), expression)
    res_expression = expression - lr.predict(PEER_factors.iloc[:,:k].as_matrix())
    return res_expression


def gaussianize(data):
    print ('Gaussianizing')
    # gaussianize input data columnwise
    g_data = np.zeros_like(data)
    n_samples = g_data.shape[1]
    for i in range(n_samples):
        x = data[:,i].copy()
        g_x = quantile_normalize_to_gaussian(x)
        g_data[:,i] = g_x
    return g_data

# Function to perform pipeline.
# parameters= {‘expression’: { ‘gaussianize’: Bool, ‘correct’: k}, ‘activations’: {‘gassianize’: Bool, ‘correct’: k } }.
def pipeline(activations, expression, parameters):

    activations_c = activations.copy()
    expression_c = expression.copy()
#     import pdb; pdb.set_trace()

    # Run parameters on expression
    if parameters['expression']['peer_correct']:
        k = parameters['expression']['peer_correct']
        expression_c = regress_out_k_peer_factors(expression_c, k)

    if parameters['expression']['sample_cov_correct']:
        k = parameters['expression']['sample_cov_correct']
        expression_c = regress_out_k_covariates(expression_c, k)

    if parameters['expression']['donor_cov_correct']:
        k = parameters['expression']['donor_cov_correct']
        expression_c = regress_out_donor_covariates(expression_c)

    # Filter to the 100 most variable expression residuals
    sorted_idx = np.argsort(expression_c.var(0))[::-1]
    expression_c = expression_c.iloc[:,sorted_idx[0:100]].as_matrix()

    if parameters['expression']['gaussianize']:
        expression_c = gaussianize(expression_c)

    # Run parameters on activations
    if parameters['activations']['peer_correct']:
        k = parameters['activations']['peer_correct']
        activations_c = regress_out_k_peer_factors(activations_c, k)

    if parameters['activations']['sample_cov_correct']:
        k = parameters['activations']['sample_cov_correct']
        activations_c = regress_out_k_covariates(activations_c, k)

    if parameters['activations']['donor_cov_correct']:
        k = parameters['activations']['donor_cov_correct']
        activations_c = regress_out_donor_covariates(activations_c)

    if parameters['activations']['gaussianize']:
#         import pdb; pdb.set_trace()
        activations_c = gaussianize(activations_c)

    print ('Computing pearson R coefficients')
    expression_c = np.array(expression_c)
    results = compute_pearsonR(activations_c, expression_c)
    return results

# Extract image activations of donors
def extract_activations():
    activations_path = GTEx_directory + '/results/DetangleInflation/activations.py'
    if os.path.exists(activations_path):
        activations = pickle.load(open(activations_path, 'rb'))
        return activations
    else:
        activations = []
        pbar = tqdm(total=len(donor_intersection))
        for d in donor_intersection:
            fn = glob.glob(GTEx_directory + '/data/features/Lung/retrained_{}*_256_l-1.hdf5'.format(d))[0]
            with h5py.File(fn) as f:
                features = f['features'].value
                agg_features = features.mean(0)
            activations.append(agg_features)
            pbar.update(1)
        pbar.close()
        activations = np.array(activations)
        pickle.dump(activations, open(activations_path, 'wb'))
        return activations

activations = extract_activations()
import pdb; pdb.set_trace()


class FeatureExploration():

    @staticmethod
    def no_correction():
        # What is inflation with no correction?
        parameters = {
            'expression': {
                'gaussianize': False,
                'peer_correct': False,
                'sample_cov_correct': False,
                'donor_cov_correct': False
                },
            'activations': {
                'gaussianize': False,
                'sample_cov_correct': False,
                'donor_cov_correct': False
                }
            }

        res = pipeline(activations, expression, parameters)
        pickle.dump(res,open(corr_dir + 'no_correction.pickle', 'wb'))

    @staticmethod
    def regress_peer():
    # How does regressing out PEER factors from expression affect inflation?
        print ('Regress Peer')
        results = []
        n_peer = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
        for k in n_peer:
            print (k)
            parameters = {
                'expression': {
                    'gaussianize': False,
                    'peer_correct': True,
                    'sample_cov_correct': False,
                    'donor_cov_correct': False
                    },
                'activations': {
                    'gaussianize': False,
                    'sample_cov_correct': False,
                    'donor_cov_correct': False
                    }
                }
            res = pipeline(activations, expression, parameters)
            results.append(res)
        pickle.dump(results, open(corr_dir + 'regress_peer.pickle', 'wb'))

    @staticmethod
    def regress_donor_covariates():
        parameters = {
            'expression': {
                'gaussianize': False,
                'peer_correct': True,
                'sample_cov_correct': False,
                'donor_cov_correct': False
                },
            'activations': {
                'gaussianize': False,
                'sample_cov_correct': False,
                'donor_cov_correct': False
                }
            }
        res = pipeline(activations, expression, parameters)
    #
    # @staticmethod
    # def regress_sample_covariates():
    #
    # @staticmethod
    # def regress_all_covariates_from_both_expression_and_activations():
