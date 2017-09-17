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
import mahotas
import scipy.stats as st
import scipy as sp



GTEx_directory = '/hps/nobackup/research/stegle/users/willj/GTEx'

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

def extract_final_layer_data(t, m, a, ps):
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



def filter_features_across_all_patchsizes(tissue, model, aggregation, M, pc_correction=False, tf_correction=False):

    """
        Computes M most varying pvalues across all patch sizes.
    """

    if pc_correction:
        print ('Correcting image features with {} expression PCs'.format(pc_correction))
    patch_sizes = [128, 256, 512, 1024, 2048, 4096]
    _, X, dIDs, tIDs, tfs, ths, t_idx = extract_final_layer_data(tissue, model, aggregation, '256')

    all_Y = []
    for ps in patch_sizes:
        Y, _, _, _, _, _, _ = extract_final_layer_data(tissue, model, aggregation, str(ps))
        if pc_correction:
            print('Correcting {} with {} PC'.format(ps, pc_correction))
            pca = PCA(n_components=pc_correction)
            pca_X = pca.fit_transform(X)
            lr = LinearRegression()
            lr.fit(pca_X, Y)
            predicted = lr.predict(pca_X)
            corrected_Y = Y - predicted
            all_Y.append(corrected_Y)
        elif tf_correction:
            print('Correcting {} with 5 TFs'.format(ps))

            Y_prime = Y[t_idx,:]
            X_prime = X[t_idx,:]
            TFs = ['SMTSISCH', 'SMNTRNRT', 'SMEXNCRT', 'SMRIN', 'SMATSSCR']
            tf_idx = [list(ths).index(x) for x in TFs]
            tf_X = tfs[:,tf_idx]
            lr = LinearRegression()
            lr.fit(tf_X, Y_prime)
            predicted = lr.predict(tf_X)
            corrected_Y = Y_prime - predicted
            all_Y.append(corrected_Y)
        else:
            all_Y.append(Y)

    concat_Y = np.concatenate(all_Y)
    most_varying_feature_idx = np.argsort(concat_Y.std(0))[-M:]

    all_filt_Y = {}
    for (i, ps) in enumerate(patch_sizes):
        all_filt_Y[ps] = all_Y[i][:, most_varying_feature_idx]



    return all_filt_Y, most_varying_feature_idx, X, dIDs, tIDs, tfs, ths, t_idx


def filter_features(Y, N):
    """
        Return top N varying image features.
    """
    most_varying_feature_idx = np.argsort(np.std(Y, axis=0))[-N:]
    filt_Y = Y[:, most_varying_feature_idx]
    return filt_Y, most_varying_feature_idx


def filter_expression(exp, tIDs, M, k):
    """
        Return top M varying transcripts, with mean expression > k, along with their transcript names.
    """
    k_threshold_idx = np.mean(exp, axis=0) > k
    filt_exp = exp[:,k_threshold_idx]
    filt_tIDs = tIDs[k_threshold_idx]

    M_varying_idx = np.argsort(np.std(filt_exp, axis=0))[-M:]

    filt_exp = filt_exp[:, M_varying_idx]
    filt_tIDs = filt_tIDs[M_varying_idx]

    return filt_exp, filt_tIDs



def compute_pearsonR(Y, X):
    """
    Perform pairwise associations between filt_features and filt_expression.
    Also computes pvalues for 3 random shuffles.
    """
    # Make sure all features are > 0
    X[X < 0] = 0

    N = Y.shape[1]
    M = X.shape[1]
    results = {}
    shuffle = ['real', 1, 2, 3]
    for sh in shuffle:
        R_mat = np.zeros((N, M))
        pvs = np.zeros((N, M))
        Y_copy = Y.copy()
        shuf_idx = list(range(Y.shape[0]))
        if sh != 'real':
            np.random.shuffle(shuf_idx)
        Y_copy = Y_copy[shuf_idx, :]

        for i in range(N):
            for j in range(M):
                R, pv = pearsonr(Y_copy[:, i], X[:, j])
                R_mat[i, j] = R
                pvs[i, j] = pv
        results['Rs_{}'.format(sh)] = R_mat
        results['pvs_{}'.format(sh)] = pvs

    return results['Rs_real'], results['pvs_real'], results['pvs_1'], \
        results['pvs_2'], results['pvs_3']


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
