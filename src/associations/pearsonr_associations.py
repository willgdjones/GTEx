import pickle
import numpy as np
import h5py
from scipy.stats import pearsonr, rankdata, truncnorm
import argparse

GTEx_directory = '/hps/nobackup/research/stegle/users/willj/GTEx'
aggregations = ['mean', 'median']
sizes = ['128', '256', '512', '1024', '2048', '4096']

parser = argparse.ArgumentParser(description='')
parser.add_argument('-n', '--N',
                    help='Number of the most varying image features to use',
                    required=True)
parser.add_argument('-m', '--M',
                    help='Number of the most varying transcripts to use',
                    required=True)
parser.add_argument('-z', '--normalize',
                    help='Normalize the image features to a truncated normal',
                    required=True)
args = vars(parser.parse_args())
N = int(args['N'])
M = int(args['M'])
normalize = int(args['normalize'])

def compute_pearsonR(image_features, expression, N, M):
    """
    Compute p-values between the top N most varying image features.
    The top M most varying transcripts + the top M most expression transcripts.
    Performs N * 2M association tests overall.
    Computes pvalues for 3 random shuffles.
    Quantile normalize values are 0 (False), 1 (True).
    """

    image_features[image_features < 0] = 0

    most_varying_feature_idx = np.argsort(np.std(image_features, axis=0))[-N:]
    most_expressed_transcript_idx = np.argsort(np.std(expression, axis=0))[-M:]
    most_varying_transcript_idx = np.argsort(np.std(expression, axis=0))[-M:]
    transcript_idx = list(most_expressed_transcript_idx) + \
        list(most_varying_transcript_idx)
    filt_image_features = image_features[:, most_varying_feature_idx]
    filt_expression = expression[:, transcript_idx]

    results = {}
    shuffle = ['real', 1, 2, 3]
    for sh in shuffle:
        R_mat = np.zeros((N, 2*M))
        pvs = np.zeros((N, 2*M))
        filt_image_features_copy = filt_image_features.copy()
        shuf_idx = list(range(filt_image_features.shape[0]))
        if sh != 'real':
            np.random.shuffle(shuf_idx)
        filt_image_features_copy = filt_image_features_copy[shuf_idx, :]

        for i in range(N):
            for j in range(2*M):
                R, pv = pearsonr(filt_expression[:, j], filt_image_features_copy[:, i])
                R_mat[i, j] = R
                pvs[i, j] = pv
        results['Rs_{}'.format(sh)] = R_mat
        results['pvs_{}'.format(sh)] = pvs

    return results['Rs_real'], results['pvs_real'], results['pvs_1'], \
        results['pvs_2'], results['pvs_3']


association_results = {}
with h5py.File(GTEx_directory + '/small_data/new_retrained_inceptionet_aggregations.hdf5', 'r') as f:
    expression = f['lung']['256']['expression'].value
    expression_IDs = f['lung']['256']['expression_IDs'].value
    for s in sizes:
        for a in aggregations:
            print('lung', s, a)
            features = f['lung'][s][a].value
            res = compute_pearsonR(features, expression, N, M)
            association_results['{}_{}_{}'.format('lung', s, a, normalize=normalize)] = res

pickle.dump(association_results, open(GTEx_directory + '/results/pvalues/association_results_N{}_M{}_norm{}.py'.format(N, M, normalize), 'wb'))
