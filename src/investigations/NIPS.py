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
import pyensembl
os.environ['PYENSEMBL_CACHE_DIR'] = '/hps/nobackup/research/stegle/users/willj/GTEx'
import statsmodels.stats.multitest as smm
from pyensembl import EnsemblRelease
from gprofiler import GProfiler
data = EnsemblRelease(77)
import multiprocess as mp
from tqdm import tqdm
# from gevent import Timeout
# from gevent import monkey
# monkey.patch_all(thread=False)
from pebble import ProcessPool
from concurrent.futures import TimeoutError



GTEx_directory = '/hps/nobackup/research/stegle/users/willj/GTEx'

os.environ['PYENSEMBL_CACHE_DIR'] = GTEx_directory

parser = argparse.ArgumentParser(description='Collection of experiments. Runs on the cluster.')
parser.add_argument('-g', '--group', help='Experiment group', required=True)
parser.add_argument('-n', '--name', help='Experiment name', required=True)
args = vars(parser.parse_args())
group = args['group']
name = args['name']


def get_gene_name(transcript):
    transcript_id = transcript.decode('utf-8').split('.')[0]
    return data.gene_name_of_gene_id(transcript_id)

def lookup_enrichment(gene_set):
    clean_gene_set = [x for x in gene_set if x is not None]
    gp = GProfiler("GTEx/wj")
    enrichment_results = gp.gprofile(clean_gene_set)
    return enrichment_results




class NIPSQuestion1():

    @staticmethod
    def calculate_variance_explained():

        os.makedirs(GTEx_directory + '/results/{}'.format(group), exist_ok=True)

        TISSUES = ['Lung', 'Artery - Tibial', 'Heart - Left Ventricle', 'Breast - Mammary Tissue', 'Brain - Cerebellum', 'Pancreas', 'Testis', 'Liver', 'Ovary', 'Stomach']
        SIZES = ['128', '256', '512', '1024', '2048', '4096']
        AGGREGATIONS = ['mean', 'median']
        MODELS = ['raw', 'retrained']

        results = {}

        for t in TISSUES:
            for a in AGGREGATIONS:
                for m in MODELS:
                    for s in SIZES:
                        key = '{}_{}_{}_{}'.format(t,a,m,s)

                        print(key)
                        image_features, expression, donorIDs, transcriptIDs, technical_factors, technical_headers, technical_idx = extract_final_layer_data(t, m, a, s)

                        print ("Calculating PCs that explain 99.9% of image feature variance")
                        pca_im = PCA(n_components=0.999)
                        pca_image_features = pca_im.fit_transform(image_features)


                        print ("Calculating PCs that explain 99.9% of expression variance")
                        pca_exp = PCA(n_components=0.999)
                        pca_expression = pca_exp.fit_transform(expression)

                        print ("Computing correlation matrix")
                        N = pca_image_features.shape[1]
                        M = pca_expression.shape[1]
                        R_matrix = np.zeros(shape=(N,M))
                        for i in range(N):
                            for j in range(M):
                                R, pv = pearsonr(pca_image_features[:,i], pca_expression[:,j])
                                R_matrix[i,j] = R

                        print ("Calculating variance of image features explained by expression ")
                        im_variance_explained = pca_im.explained_variance_
                        exp_variance_explained = pca_exp.explained_variance_

                        #sum(R_matrix[k,:]) ~ 1 for all k.
                        componentwise_variance_explained = []
                        for i in range(len(im_variance_explained)):
                            component_variance_explained = im_variance_explained[i] * sum(R_matrix[:,i]**2)
                            componentwise_variance_explained.append(component_variance_explained)

                        total = sum(componentwise_variance_explained)
                        frac = total / sum(im_variance_explained)

                        print (total, frac)

                        results[key] = [total, frac]


        pickle.dump([results], open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))




class NIPSQuestion2():
    @staticmethod
    def shared_variability():

        os.makedirs(GTEx_directory + '/results/{}'.format(group), exist_ok=True)

        TISSUES = ['Lung', 'Artery - Tibial', 'Heart - Left Ventricle', 'Breast - Mammary Tissue', 'Brain - Cerebellum', 'Pancreas', 'Testis', 'Liver', 'Ovary', 'Stomach']
        SIZES = ['128', '256', '512', '1024', '2048', '4096']
        AGGREGATIONS = ['mean', 'median']
        MODELS = ['raw', 'retrained']
        TFs = ['SMTSISCH', 'SMNTRNRT', 'SMEXNCRT', 'SMRIN', 'SMATSSCR']

        results = {}

        for t in TISSUES:
            for a in AGGREGATIONS:
                for m in MODELS:
                    for s in SIZES:
                        key = '{}_{}_{}_{}'.format(t,a,m,s)

                        print (key)

                        Y, X, dIDs, tIDs, tfs, ths, t_idx = extract_final_layer_data(t, m, a, s)
                        Y_prime = Y[t_idx,:]
                        X_prime = X[t_idx,:]
                        tf_idx = [list(ths).index(x) for x in TFs]

                        # Take log of SMTSISH
                        tfs[list(ths).index('SMTSISCH')] = np.log2(tfs[list(ths).index('SMTSISCH')] + 1)
                        tf_predictors = tfs

                        lr_X = LinearRegression()


                        # Regress out effects from X
                        lr_X.fit(tf_predictors, X_prime)
                        X_predicted = lr_X.predict(tf_predictors)
                        corrected_X = X_prime - X_predicted


                        lr_Y = LinearRegression()

                        # Regress out effects from X
                        lr_Y.fit(tf_predictors, Y_prime)
                        Y_predicted = lr_Y.predict(tf_predictors)
                        corrected_Y = Y_prime - Y_predicted

                        print ("Calculating PCs that explain 99.9% of corrected image feature variance")
                        pca_Y = PCA(n_components=0.999)
                        pca_corrected_Y = pca_Y.fit_transform(corrected_Y)


                        print ("Calculating PCs that explain 99.9% of expression variance")
                        pca_X = PCA(n_components=0.999)
                        pca_corrected_X = pca_X.fit_transform(corrected_X)

                        print ("Computing correlation matrix")
                        N = pca_corrected_Y.shape[1]
                        M = pca_corrected_X.shape[1]
                        R_matrix = np.zeros(shape=(N,M))
                        for i in range(N):
                            for j in range(M):
                                R, pv = pearsonr(pca_corrected_Y[:,i], pca_corrected_X[:,j])
                                R_matrix[i,j] = R

                        print ("Calculating variance of image features explained by expression ")
                        Y_variance_explained = pca_Y.explained_variance_
                        X_variance_explained = pca_X.explained_variance_

                        componentwise_variance_explained = []
                        for i in range(len(Y_variance_explained)):
                            component_variance_explained = Y_variance_explained[i] * sum(R_matrix[:,i]**2)
                            componentwise_variance_explained.append(component_variance_explained)

                        total = sum(componentwise_variance_explained)
                        frac = total / sum(Y_variance_explained)

                        print (total, frac)

                        results[key] = [total, frac]


        pickle.dump(results, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))




class NIPSQuestion3A():


    @staticmethod
    def gene_expression_variability():

        os.makedirs(GTEx_directory + '/results/{}'.format(group), exist_ok=True)
        final_results = {}

        Y, X, dIDs, tIDs, tfs, ths, t_idx = extract_final_layer_data('Lung', 'retrained', 'mean', '256')
        tfs[list(ths).index('SMTSISCH')] = np.log2(tfs[list(ths).index('SMTSISCH')] + 1)

        tf_X = X[t_idx,:]

        def pbar_update(x):
            pbar.update(1)

        def calculate_tf_enrichment(th):

            i = list(ths).index(th)
            tf = tfs[:, i].reshape(-1,1)


            Rs = []
            pvs = []
            for t in range(tf_X.shape[1]):
                R, pv = pearsonr(tf.flatten(), tf_X[:,t])
                Rs.append(R)
                pvs.append(pv)


            significant_transcripts = tIDs[smm.multipletests(pvs, method='bonferroni',alpha=0.01)[0]]

            significant_genes = []
            for t in significant_transcripts:
                try:
                    g = get_gene_name(t)
                except ValueError:
                    g = None
                significant_genes.append(g)

            enrichments = lookup_enrichment(significant_genes)
            return enrichments


        # pool = mp.Pool(processes=2)
        # pbar = tqdm(total=len(list(ths)))


        print ("Looking up gene enrichments")
        results = [apply(calculate_tf_enrichment, args=(th,), callback=pbar_update) for th in list(ths)]
        enrichment_results = [p.get() for p in results]

        pickle.dump(enrichment_results, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))

class NIPSQuestion5():
    @staticmethod
    def define_genetic_subset_snps():
        os.makedirs(GTEx_directory + '/results/{}'.format(group), exist_ok=True)
        print ("Loading genotype data")

        Y, X, G, dIDs, tIDs, gIDs, tfs, ths, t_idx = extract_final_layer_data('Lung', 'retrained', 'mean', '256', genotypes=True)

        association_results, filt_transcriptIDs = pickle.load(open(GTEx_directory + '/intermediate_results/TFCorrectedFeatureAssociations/compute_pvalues.pickle', 'rb'))

        print ("Calculating significant transcripts")
        significant_indicies = [smm.multipletests(association_results['Lung_mean_retrained_256'][1][i,:],method='bonferroni',alpha=0.01)[0] for i in range(1024)]
        significant_counts = [sum(x) for x in significant_indicies]
        significant_transcripts = [filt_transcriptIDs[x] for x in significant_indicies]



        print ("Translating into significant genes")
        significant_genes = []
        for (i, feature_transcripts) in enumerate(significant_transcripts):
            if i % 100 == 0:
                print ("Gene set ", i)
            genes = []
            for t in feature_transcripts:
                try:
                    g = get_gene_name(t)
                except ValueError:
                    g = None

                genes.append(g)
            significant_genes.append(genes)

        all_intervals = []
        for gene_set in significant_genes:
            w = 5000
            gene_set_intervals = []
            if not None in gene_set:

                for gene in gene_set:
                    gene_obj = data.genes_by_name(gene)[0]
                    interval_start = gene_obj.start - w
                    interval_end = gene_obj.end + w
                    chrom = gene_obj.contig
                    interval = (interval_start, interval_end, chrom)
                    gene_set_intervals.append(interval)
            else:
                interval = None
                gene_set_intervals.append(interval)


            all_intervals.append(gene_set_intervals)

        from tqdm import tqdm
        pbar = tqdm(total=len(all_intervals))

        g_idx = np.array(range(gIDs.shape[1]))



        pbar = tqdm(total=len(all_intervals))
        # pbar = tqdm(total=10)


        def get_snp_sets(interval_set):
            snp_sets = []
            if interval_set != [] and interval_set[0] is not None:
                for interval in interval_set:
                    start = interval[0]
                    end = interval[1]
                    chrom = interval[2]

                    chrom_idx = gIDs[0] == chrom.encode('utf-8')
                    chrom_region = gIDs[:,chrom_idx]

                    snp_idx = np.bitwise_and(chrom_region[1,:].astype(np.int64) > start, chrom_region[1,:].astype(np.int64) < end)
                    snp_set = g_idx[chrom_idx][snp_idx]

                    snp_sets.extend(snp_set)
                # pbar.update(1)
                return snp_sets
            else:
                # pbar.update(1)
                return snp_sets


            # all_snp_sets = []
            # while True:
            #     try:
            #         result = next(future_results)
            #         all_snp_sets.append(result)
            #         pbar.update(1)
            #     except StopIteration:
            #         break
            #     except TimeoutError as error:
            #         all_snp_sets.append(None)
            #         pbar.update(1)
            #         print("function took longer than %d seconds" % error.args[1])
            #     except ProcessExpired as error:
            #         all_snp_sets.append(None)
            #         pbar.update(1)
            #         print("%s. Exit code: %d" % (error, error.exitcode))
            #     except Exception as error:
            #         all_snp_sets.append(None)
            #         print("function raised %s" % error)
            #         print(error)  # Python's traceback of remote process

        all_snp_sets = []
        for interval_set in all_intervals:
            snp_sets = get_snp_sets(interval_set)
            all_snp_sets.append(snp_sets)
            pbar.update(1)

        pickle.dump(all_snp_sets, open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))


    @staticmethod
    def perform_association_tests():
        from scipy.stats import norm

        all_snp_sets = pickle.load(open(GTEx_directory + '/results/NIPSQuestion5/define_genetic_subset_snps.pickle', 'rb'))
        print ("Loading genotype data")
        Y, X, G, dIDs, tIDs, gIDs, tfs, ths, t_idx = extract_final_layer_data('Lung', 'retrained', 'mean', '256', genotypes=True)

        def quantile_normalize_using_target(x, target):
            """
            Both `x` and `target` are numpy arrays of equal lengths.
            """

            target_sorted = np.sort(target)

            return target_sorted[x.argsort().argsort()]

        def normalize_feature(original_feature):
            mu, std = norm.fit(original_feature)
            target = [np.random.normal()*std + mu for i in range(271)]
            result = quantile_normalize_using_target(original_feature, target)
            return result


        print ("Normalising data")

        n_Y = np.zeros_like(Y)
        for i in range(1024):
            original_feature = Y[:,i]
            normalized_feature = normalize_feature(original_feature)
            n_Y[:,i] = normalized_feature


        all_snps = []
        for set_set in all_snp_sets:
            all_snps.extend(set_set)

        all_snps_flat = list(set(all_snps))

        G_candidates = G[:,all_snps_flat]
        G_candidates[G_candidates == 255] = 0

        from sklearn.preprocessing import normalize
        G_normalized = normalize(G_candidates)
        K = np.dot(G_normalized, G_normalized.T)

        from limix.qtl import LMM

        print ("Performing associations")

        lmm = LMM(np.asarray(G_candidates, np.float64), np.asarray(n_Y, np.float64), np.asarray(K, np.float64))
        pvalues = lmm.getPv()
        betas = lmm.getBetaSNP()

        os.makedirs(GTEx_directory + '/intermediate_results/{}'.format(group), exist_ok=True)
        pickle.dump([pvalues, betas], open(GTEx_directory + '/intermediate_results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))

    @staticmethod
    def top_association_results():
        [pvalues, betas] = pickle.load(open(GTEx_directory + '/intermediate_results/{group}/perform_association_tests.pickle'.format(group=group), 'rb'))
        Y, X, G, dIDs, tIDs, gIDs, tfs, ths, t_idx = extract_final_layer_data('Lung', 'retrained', 'mean', '256', genotypes=True)
        all_snp_sets = pickle.load(open(GTEx_directory + '/results/NIPSQuestion5/define_genetic_subset_snps.pickle', 'rb'))

        all_snps = []
        for set_set in all_snp_sets:
            all_snps.extend(set_set)

        all_snps_flat = list(set(all_snps))

        gID_candidates = gIDs[:,all_snps_flat]
        G_candidates = G[:,all_snps_flat]
        G_candidates[G_candidates == 255] = 0

        [pvalues, betas] = pickle.load(open(GTEx_directory + '/intermediate_results/NIPSQuestion5//perform_association_tests.pickle', 'rb'))
        flat_pvalues = np.array(pvalues).flatten()
        flat_betas = np.array(betas).flatten()

        unique_sorted_pvalues = np.unique(flat_pvalues)
        unique_sorted_betas = np.unique(flat_betas)

        gIDs_candidates = gIDs[:,all_snps_flat]

        N = 50
        pbar1 = tqdm(total=N)
        pbar2 = tqdm(total=N)

        top_pvs = []
        for pv in unique_sorted_pvalues[0:N]:
            indicies = np.argwhere(pvalues == pv)[0]
            g = G_candidates[:, indicies[1]]
            y = Y[:, indicies[0]]
            gID = gIDs_candidates[:, indicies[1]]
            top_pvs.append((pv, indicies, g, gID, y))
            pbar1.update(1)

        top_betas = []
        for b in unique_sorted_betas[0:N]:
            indicies = np.argwhere(betas == b)[0]
            g = G_candidates[:, indicies[1]]
            y = Y[:, indicies[0]]
            gID = gIDs_candidates[:, indicies[1]]
            top_betas.append((b, indicies, g, gID, y))
            pbar2.update(1)

        pickle.dump([top_pvs, top_betas], open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'wb'))




















if __name__ == '__main__':
    eval(group + '().' + name + '()')
