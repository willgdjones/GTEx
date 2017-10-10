import os
import h5py
from matplotlib.colors import Normalize
import gzip
import pandas as pd
import numpy as np
from matplotlib import cbook
from numpy import ma
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import cv2
# import mahotas
import scipy.stats as st
import scipy as sp
from tqdm import tqdm
from pebble import ProcessPool, ProcessExpired
from concurrent.futures import TimeoutError
from pyensembl import EnsemblRelease
data = EnsemblRelease(77)


GTEx_directory = '/hps/nobackup/research/stegle/users/willj/GTEx'
os.environ['PYENSEMBL_CACHE_DIR'] = GTEx_directory

class MidPointNorm(Normalize):

    """
        Ensures that heatmap colour bars are zero centered.
    """

    def __init__(self, midpoint=0, vmin=None, vmax=None, clip=False):
        Normalize.__init__(self,vmin, vmax, clip)
        self.midpoint = midpoint

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        self.autoscale_None(result)
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if not (vmin < midpoint < vmax):
            raise ValueError("midpoint must be between maxvalue and minvalue.")
        elif vmin == vmax:
            result.fill(0) # Or should it be all masked? Or 0.5?
        elif vmin > vmax:
            raise ValueError("maxvalue must be bigger than minvalue")
        else:
            vmin = float(vmin)
            vmax = float(vmax)
            if clip:
                mask = ma.getmask(result)
                result = ma.array(np.clip(result.filled(vmax), vmin, vmax),
                                  mask=mask)

            # ma division is very slow; we can take a shortcut
            resdat = result.data

            #First scale to -1 to 1 range, than to from 0 to 1.
            resdat -= midpoint
            resdat[resdat>0] /= abs(vmax - midpoint)
            resdat[resdat<0] /= abs(vmin - midpoint)

            resdat /= 2.
            resdat += 0.5
            result = ma.array(resdat, mask=result.mask, copy=False)

        if is_scalar:
            result = result[0]
        return result

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until scaled")
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if cbook.iterable(value):
            val = ma.asarray(value)
            val = 2 * (val-0.5)
            val[val>0]  *= abs(vmax - midpoint)
            val[val<0] *= abs(vmin - midpoint)
            val += midpoint
            return val
        else:
            val = 2 * (val - 0.5)
            if val < 0:
                return  val*abs(vmin-midpoint) + midpoint
            else:
                return  val*abs(vmax-midpoint) + midpoint

def extract_final_layer_data(t, m, a, ps, genotypes=False):
    with h5py.File(GTEx_directory +
                '/data/h5py/aggregated_features.h5py', 'r') as f:
        X = f[t]['ordered_expression'].value

        tIDs = f[t]['transcriptIDs'].value
        dIDs = f[t]['donorIDs'].value
        tfs, ths, t_idx = \
            get_technical_factors(t, dIDs)
        size_group = f[t]['-1'][ps]
        Y = size_group[m][a]['ordered_aggregated_features'].value
        Y[Y < 0] = 0
        if genotypes:

            G = f[t]['ordered_genotypes'].value
            gIDs = f[t]['genotype_locations'].value
            return Y, X, G, dIDs, tIDs, gIDs, \
                tfs, ths, t_idx
        else:

            return Y, X, dIDs, tIDs, \
                tfs, ths, t_idx


def extract_mid_layer_data(t, l, ca, m, a, ps):
    with h5py.File(GTEx_directory +
                '/data/h5py/aggregated_features.h5py', 'r') as f:
        expression = f[t]['ordered_expression'].value
        transcriptIDs = f[t]['transcriptIDs'].value
        donorIDs = f[t]['donorIDs'].value
        technical_factors, technical_headers, technical_idx = \
            get_technical_factors(t, donorIDs)
        size_group = f[t][l][ca][ps]
        features = size_group[m][a]['ordered_aggregated_features'].value
        features[features < 0] = 0
        return features, expression, donorIDs, transcriptIDs, \
            technical_factors, technical_headers, technical_idx


def get_technical_factors(tissue, donorIDs):

    phenotype_filepath = '/nfs/research2/stegle/stegle_secure/GTEx/download/49139/PhenoGenotypeFiles/RootStudyConsentSet_phs000424.GTEx.v6.p1.c1.GRU/PhenotypeFiles/phs000424.v6.pht002743.v6.p1.c1.GTEx_Sample_Attributes.GRU.txt.gz'
    with gzip.open(phenotype_filepath, 'rb') as f:
        g = f.read().splitlines()
        phenotype_array = [str(x, 'utf-8').split('\t') for x in g if not str(x, 'utf-8').startswith('#')]
        phenotype_array = phenotype_array[1:]
        phenotype_df = pd.DataFrame(phenotype_array)
        phenotype_df.columns = phenotype_df.iloc[0]
        phenotype_df = phenotype_df[1:]

    tissue_df = phenotype_df[phenotype_df['SMTSD'] == tissue]
    donorIDs = [x.decode('utf-8') for x in donorIDs]
    phenotype_donorIDs = [x.split('-')[1] for x in tissue_df['SAMPID']]
    phenotype_idx = [phenotype_donorIDs.index(ID) for ID in donorIDs]
    tissue_df = phenotype_df[phenotype_df['SMTSD'] == tissue]
    tissue_df = tissue_df.iloc[phenotype_idx, :]
    SMCENTER_dummy = pd.get_dummies(tissue_df['SMCENTER'])
    for d in SMCENTER_dummy.columns:
        tissue_df['SMCENTER_' + d] = SMCENTER_dummy[d]

    clean_tissue_df = pd.DataFrame()
    for col in tissue_df.columns:
        clean_factor = pd.to_numeric(tissue_df[col], errors='coerce')
        clean_tissue_df[col] = clean_factor
    clean_tissue_df = clean_tissue_df.dropna(how='all', axis=1)
    technical_idx = np.array(clean_tissue_df.isnull().sum(axis=1) == 0)
    clean_tissue_df = clean_tissue_df.dropna(how='any', axis=0)
    technical_factors, technical_headers = \
        np.array(clean_tissue_df), clean_tissue_df.columns

    technical_headers = technical_headers[technical_factors.std(0) > 0]
    technical_factors = technical_factors[:,technical_factors.std(0) > 0]

    return technical_factors, technical_headers, technical_idx





def filter_and_correct_expression_and_image_features(tissue, model, aggregation, patch_size, M, k, pc_correction=False, tf_correction=False):

    """
        Computes M most varying pvalues across all patch sizes.
        - Filters to the top M most varying genes that have mean expression > k.

        Optional:
        - Performs PC correction - regresses out effect of first x PCs from image features, and substracts the first x PCs from the expression matrix.
        - Performs TF correction - regresses out effect of five PCs from both the image features, and expression.
    """




    # Filter expression
    Y, X, dIDs, tIDs, tfs, ths, t_idx = extract_final_layer_data(tissue, model, aggregation, patch_size)
    filt_X, filt_tIDs, final_exp_idx = filter_expression(X, tIDs, M, k)



    if pc_correction:
        print ('Correcting with {} expression PCs'.format(pc_correction))
        pca = PCA(n_components=pc_correction)


        pca_predictors = pca.fit_transform(filt_X)

        # Correct Y
        lr = LinearRegression()
        lr.fit(pca_predictors, Y)
        predicted_Y = lr.predict(pca_predictors)
        corrected_Y = Y - predicted_Y

        # Correct X
        projected_filt_X = np.dot(pca_predictors,pca.components_)
        corrected_filt_X = filt_X - projected_filt_X

        # Set as return variables
        final_X = corrected_filt_X
        final_Y = corrected_Y

    elif tf_correction:
        print('Correcting with all technical factors')
        tf_Y = Y[t_idx,:]
        tf_filt_X = filt_X[t_idx,:]

        tfs[list(ths).index('SMTSISCH')] = np.log2(tfs[list(ths).index('SMTSISCH')] + 1)
        tf_predictors = tfs

        #Correct Y
        lr_Y = LinearRegression()
        lr_Y.fit(tf_predictors, tf_Y)
        tf_Y_predicted = lr_Y.predict(tf_predictors)
        corrected_tf_Y = tf_Y - tf_Y_predicted

        #Correct X
        lr_X = LinearRegression()
        lr_X.fit(tf_predictors, tf_filt_X)
        tf_filt_X_predicted = lr_X.predict(tf_predictors)
        corrected_tf_filt_X = tf_filt_X - tf_filt_X_predicted

        # Set as return variables
        final_X = corrected_tf_filt_X
        final_Y = corrected_tf_Y
    else:
        # Set unmodified values as return variables
        final_X = filt_X
        final_Y = Y

    return final_Y, final_X, dIDs, filt_tIDs, tfs, ths, t_idx


def filter_features(Y, N):
    """
        Return top N varying image features.
    """
    most_varying_feature_idx = np.argsort(np.std(Y, axis=0))[-N:]
    filt_Y = Y[:, most_varying_feature_idx]
    return filt_Y, most_varying_feature_idx


def filter_expression(X, tIDs, M, k):
    """
        Return top M varying transcripts, with mean expression > k, along with their transcript names.
    """
    k_threshold_idx = np.mean(X, axis=0) > k
    M_varying_idx = np.argsort(np.std(X[:,k_threshold_idx], axis=0))[-M:]
    idx = np.array(list(range(X.shape[1])))
    final_exp_idx = idx[k_threshold_idx][M_varying_idx]

    filt_X = X[:, final_exp_idx]
    filt_tIDs = tIDs[final_exp_idx]

    return filt_X, filt_tIDs, final_exp_idx




def compute_pearsonR(Y, X, parallel=False):
    """
    Perform pairwise associations between filt_features and filt_expression.
    Also computes pvalues for 3 random shuffles.
    """
    # Make sure all features are > 0
    X[X < 0] = 0

    N = Y.shape[1]
    M = X.shape[1]

    if parallel:
        print('Computing in parallel')

    results = {}
    shuffle = ['real', 'shuffle']
    for sh in shuffle:
        print ("Shuffle: {}".format(sh))

        Y_copy = Y.copy()
        shuf_idx = list(range(Y.shape[0]))
        if sh != 'real':
            np.random.shuffle(shuf_idx)
        Y_copy = Y_copy[shuf_idx, :]

        if parallel:

            pbar = tqdm(total=N*M)

            def perform_pearsonr(idx):
                i, j = idx
                R, pv = pearsonr(Y_copy[:, i], X[:, j])
                pbar.update(1)

                return R, pv

            indicies = []
            for i in range(N):
                for j in range(M):
                    idx = (i,j)
                    indicies.append(idx)

            import pathos
            import time

            pool = pathos.pools.ProcessPool(node=32)
            results = pool.map(perform_pearsonr, indicies)
            pbar.close()
            R_mat = np.array([x[0] for x in results]).reshape(N,M)
            pvs = np.array([x[1] for x in parallel_results]).reshape(N,M)

        else:
            pbar = tqdm(total=N*M)
            R_mat = np.zeros((N, M))
            pvs = np.zeros((N, M))
            for i in range(N):
                for j in range(M):
                    R, pv = pearsonr(Y_copy[:, i], X[:, j])
                    R_mat[i, j] = R
                    pvs[i, j] = pv
                    pbar.update(1)
            pbar.close()


        results['Rs_{}'.format(sh)] = R_mat
        results['pvs_{}'.format(sh)] = pvs

    return results['Rs_real'], results['pvs_real'], results['pvs_shuffle']


def create_tissue_boundary(ID, tissue, patchsize):
    from openslide import open_slide

    image_filepath = os.path.join(GTEx_directory, 'data', 'raw', tissue, ID + '.svs')

    image_slide = open_slide(image_filepath)
    toplevel = image_slide.level_count - 1
    topdim = image_slide.level_dimensions[-1]
    topdownsample = image_slide.level_downsamples[-1]
    topdownsampleint = int(topdownsample)

    toplevelslide = image_slide.read_region((0, 0), toplevel, topdim)
    toplevelslide = np.array(toplevelslide)
    toplevelslide = toplevelslide[:, :, 0:3]
    slide = toplevelslide

    blurredslide = cv2.GaussianBlur(slide, (51, 51), 0)
    blurredslide = cv2.cvtColor(blurredslide, cv2.COLOR_BGR2GRAY)
    T_otsu = mahotas.otsu(blurredslide)

    mask = np.zeros_like(slide)
    mask = mask[:, :, 0]
    mask[blurredslide < T_otsu] = 255


    downsampledpatchsize = patchsize / topdownsampleint
    xlimit = int(topdim[1] / downsampledpatchsize)
    ylimit = int(topdim[0] / downsampledpatchsize)


    # Find downsampled coords
    coords = []
    for i in range(xlimit):
        for j in range(ylimit):
            x = int(downsampledpatchsize/2 + i*downsampledpatchsize)
            y = int(downsampledpatchsize/2 + j*downsampledpatchsize)
            coords.append((x, y))

    # Find coords in downsampled mask
    mask_coords = []
    for c in coords:
        x = c[0]
        y = c[1]
        if mask[x, y] > 0:
            mask_coords.append(c)

    slidemarkings = slide.copy()
    for c in mask_coords:
        x = c[0]
        y = c[1]
        slidemarkings[x-3:x+3, y-3:y+3] = [0, 0, 255]

    return slide, mask, slidemarkings


def top5_bottom5_image(tissue, model, patchsize, feature):

    """
        Displays thumbnails of the top 5 and bottom 5 images that activate a
        given image features at a specific patchsize
    """

    from openslide import open_slide
    features, expression, donorIDs, transcriptIDs, technical_factors, technical_headers, technical_idx = extract_final_layer_data(tissue, model, 'mean', patchsize)

    sorted_idx = np.argsort(features[:,feature - 1])
    donorIDs_ordered = donorIDs[sorted_idx]

    tissue_filepath = os.path.join(GTEx_directory,'data','raw',tissue)

    LungGTExIDs = os.listdir(tissue_filepath)
    LungdonorIDs = [x.split('.')[0].split('-')[1] for x in LungGTExIDs]

    ordered_GTExIDs = np.array(LungGTExIDs)[[LungdonorIDs.index(x.decode('utf-8')) for x in donorIDs_ordered]]

    topIDs = ordered_GTExIDs[-5:]
    bottomIDs = ordered_GTExIDs[:5]

    top_five_images = []
    bottom_five_images = []

    for (k,ID) in enumerate(topIDs):
        image_filepath = os.path.join(GTEx_directory,'data','raw','Lung', ID)
        slide = open_slide(image_filepath)
        x = slide.get_thumbnail(size=(400,400))
        top_five_images.append(x)


    for (k,ID) in enumerate(bottomIDs):
        image_filepath = os.path.join(GTEx_directory,'data','raw','Lung', ID)
        slide = open_slide(image_filepath)
        x = slide.get_thumbnail(size=(400,400))
        bottom_five_images.append(x)

    return top_five_images, bottom_five_images


def estimate_lambda(pv):
    """estimate lambda form a set of PV"""
    LOD2 = sp.median(st.chi2.isf(pv, 1))
    null_median = st.chi2.median(1)
    L = (LOD2 / null_median)
    return L


def display_tissue_feature_gradient(feature, tissue):
    from openslide import open_slide
    features, expression, donorIDs, transcriptIDs, technical_factors, technical_headers, technical_idx = extract_final_layer_data(tissue, 'retrained', 'mean', '256')
    sorted_idx = np.argsort(features[:,feature - 1])
    donorIDs_ordered = donorIDs[sorted_idx]
    gradient_IDs = [donorIDs_ordered[20*i] for i in range(13)]

    tissue_filepath = os.path.join(GTEx_directory,'data','raw',tissue)
    LungGTExIDs = os.listdir(tissue_filepath)
    LungdonorIDs = [x.split('.')[0].split('-')[1] for x in LungGTExIDs]

    ordered_GTExIDs = np.array(LungGTExIDs)[[LungdonorIDs.index(x.decode('utf-8')) for x in donorIDs_ordered]]

    thumbnails = []
    pbar = tqdm(total=len(ordered_GTExIDs))
    for (k,ID) in enumerate(ordered_GTExIDs):
        image_filepath = os.path.join(GTEx_directory,'data','raw','Lung', ID)
        slide = open_slide(image_filepath)
        thumbnail = slide.get_thumbnail(size=(400,400))
        feature_value = features[:,feature - 1][sorted_idx[k]]
        thumbnails.append((thumbnail, feature_value))
        pbar.update(1)

    return thumbnails


def get_gene_name(transcript):
    transcript_id = transcript.decode('utf-8').split('.')[0]
    try:
        gene_name = data.gene_name_of_gene_id(transcript_id)
    except:
        gene_name = transcript_id
    return gene_name
