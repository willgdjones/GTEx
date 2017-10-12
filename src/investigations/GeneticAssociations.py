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
from pebble import ProcessPool
import pathos
from concurrent.futures import TimeoutError



GTEx_directory = '/hps/nobackup/research/stegle/users/willj/GTEx'

os.environ['PYENSEMBL_CACHE_DIR'] = GTEx_directory

parser = argparse.ArgumentParser(description='Collection of experiments. Runs on the cluster.')
parser.add_argument('-g', '--group', help='Experiment group', required=True)
parser.add_argument('-n', '--name', help='Experiment name', required=True)
parser.add_argument('-p', '--params', help='Parameters')
args = vars(parser.parse_args())
group = args['group']
name = args['name']
parameter_key = args['params']

class GeneticAssociations():

    @staticmethod
    def define_genetic_subset_snps():
        os.makedirs(GTEx_directory + '/results/{}'.format(group), exist_ok=True)


        t, a, m, s = parameter_key.split('_')

        print ("Loading genotype data for {}".format(parameter_key))

        Y, X, G, dIDs, tIDs, gIDs, tfs, ths, t_idx = extract_final_layer_data(t, m, a, s, genotypes=True)

        association_results, filt_transcriptIDs = pickle.load(open(GTEx_directory + '/intermediate_results/TFCorrectedFeatureAssociations/compute_pvalues_{key}.pickle'.format(key=parameter_key), 'rb'))

        print ("Calculating significant transcripts")
        significant_indicies = [smm.multipletests(association_results[1][i,:],method='bonferroni',alpha=0.001)[0] for i in range(1024)]
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
            w = 1000
            gene_set_intervals = []
            if not None in gene_set:

                for gene in gene_set:
                    try:
                        gene_obj = data.genes_by_name(gene)[0]
                        interval_start = gene_obj.start - w
                        interval_end = gene_obj.end + w
                        chrom = gene_obj.contig
                        interval = (interval_start, interval_end, chrom)
                        gene_set_intervals.append(interval)
                    except:
                        gene_set_intervals.append(None)


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
                    if interval == None:
                        continue
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


        # pool = pathos.pools.ProcessPool(node=8)

        # all_snp_sets = pool.imap(get_snp_sets, all_intervals)

        # for i, _ in enumerate(all_snp_sets):
        #     pbar.update(1)
        # import pdb; pdb.set_trace()

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

        pickle.dump(all_snp_sets, open(GTEx_directory + '/intermediate_results/{group}/{name}_{key}.pickle'.format(group=group, name=name, key=parameter_key), 'wb'))


    @staticmethod
    def perform_association_tests():
        

        t, a, m, s = parameter_key.split('_')

        association_results, filt_transcriptIDs = pickle.load(open(GTEx_directory + '/intermediate_results/TFCorrectedFeatureAssociations/compute_pvalues_{key}.pickle'.format(key=parameter_key), 'rb'))
        all_snp_sets = pickle.load(open(GTEx_directory + '/intermediate_results/GeneticAssociations/define_genetic_subset_snps_{key}.pickle'.format(key=parameter_key), 'rb'))
        print ("Loading genotype data")
        Y, X, G, dIDs, tIDs, gIDs, tfs, ths, t_idx = extract_final_layer_data(t, m, a, s, genotypes=True)

        print ("Calculating significant transcripts")
        significant_indicies = [smm.multipletests(association_results[1][i,:],method='bonferroni',alpha=0.001)[0] for i in range(1024)]
        significant_counts = [sum(x) for x in significant_indicies]
        significant_transcripts = [filt_transcriptIDs[x] for x in significant_indicies]


        feature_idx = np.array(significant_counts) > 0


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

        lmm = LMM(np.asarray(G_candidates, np.float64), np.asarray(n_Y[:,feature_idx], np.float64), np.asarray(K, np.float64))
        pvalues = lmm.getPv()
        betas = lmm.getBetaSNP()

        os.makedirs(GTEx_directory + '/intermediate_results/{}'.format(group), exist_ok=True)
        pickle.dump([pvalues, betas, feature_idx], open(GTEx_directory + '/intermediate_results/{group}/{name}_{key}.pickle'.format(group=group, name=name, key=parameter_key), 'wb'))

    @staticmethod
    def top_association_results():
        [pvalues, betas, feature_idx] = pickle.load(open(GTEx_directory + '/intermediate_results/{group}/perform_association_tests_{key}.pickle'.format(group=group, key=parameter_key), 'rb'))

        t, a, m, s = parameter_key.split('_')

        Y, X, G, dIDs, tIDs, gIDs, tfs, ths, t_idx = extract_final_layer_data(t, m, a, s, genotypes=True)
        all_snp_sets = pickle.load(open(GTEx_directory + '/results/GeneticAssociations/define_genetic_subset_snps_{key}.pickle'.format(key=parameter_key), 'rb'))

        all_snps = []
        for set_set in all_snp_sets:
            all_snps.extend(set_set)

        all_snps_flat = list(set(all_snps))

        gID_candidates = gIDs[:,all_snps_flat]
        G_candidates = G[:,all_snps_flat]
        G_candidates[G_candidates == 255] = 0
        flat_pvalues = np.array(pvalues).flatten()
        flat_betas = np.array(betas).flatten()


        association_resultsbh01 = smm.multipletests(flat_pvalues, method='fdr_bh',alpha=0.01)
        import pdb; pdb.set_trace()

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

        pickle.dump([top_pvs, top_betas, feature_idx], open(GTEx_directory + '/results/{group}/{name}_{key}.pickle'.format(group=group, name=name, key=parameter_key), 'wb'))




if __name__ == '__main__':
    eval(group + '().' + name + '()')
