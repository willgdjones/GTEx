import h5py
from matplotlib.colors import Normalize
import gzip
import pandas as pd
import numpy as np
from matplotlib import cbook
from numpy import ma

GTEx_directory = '/hps/nobackup/research/stegle/users/willj/GTEx'


def extract_final_layer_data(t, m, a, ps):
    with h5py.File(GTEx_directory +
                '/data/h5py/aggregated_features.h5py', 'r') as f:
        expression = f[t]['ordered_expression'].value
        transcriptIDs = f[t]['transcriptIDs'].value
        donorIDs = f[t]['donorIDs'].value
        technical_factors, technical_headers, technical_idx = \
            get_technical_factors(t, donorIDs)
        size_group = f[t]['-1'][ps]
        features = size_group[m][a]['ordered_aggregated_features'].value
        features[features < 0] = 0
        return features, expression, donorIDs, transcriptIDs, \
            technical_factors, technical_headers, technical_idx


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


def filter_features(features, N):
    most_varying_feature_idx = np.argsort(np.std(features, axis=0))[-N:]
    filt_features = features[:, most_varying_feature_idx]
    return filt_features, most_varying_feature_idx


def filter_expression(expression, transcriptIDs, M):
    most_expressed_transcript_idx = np.argsort(np.mean(expression, axis=0))[-M:]
    most_varying_transcript_idx = np.argsort(np.std(expression, axis=0))[-M:]
    transcript_idx = list(most_expressed_transcript_idx) + \
        list(most_varying_transcript_idx)
    filt_expression = expression[:, transcript_idx]
    filt_transcriptIDs = transcriptIDs[transcript_idx]
    return filt_expression, filt_transcriptIDs, transcript_idx


class MidPointNorm(Normalize):
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
