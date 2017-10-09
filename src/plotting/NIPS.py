import os
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import h5py
import argparse
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pylab as PL
import matplotlib
from matplotlib import cbook
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
sys.path.insert(0, os.getcwd())
from src.utils.helpers import *
from tabulate import tabulate


GTEx_directory = '.'

parser = argparse.ArgumentParser(description='Collection of plotting results. Runs on local computer.')
parser.add_argument('-g', '--group', help='Plotting group', required=True)
parser.add_argument('-n', '--name', help='Plotting name', required=True)
args = vars(parser.parse_args())
group = args['group']
name = args['name']



class NIPS():

    @staticmethod
    def compare_variance_explained():
        import seaborn as sns
        sns.set_style("dark")


        SIZES = ['128', '256', '512', '1024', '2048', '4096']

        raw_results = pickle.load(open(GTEx_directory + '/results/NIPSQuestion1/calculate_variance_explained.pickle', 'rb'))[0]
        corrected_results = pickle.load(open(GTEx_directory + '/results/NIPSQuestion2/shared_variability.pickle', 'rb'))

        ordered_x = []
        ordered_y = []
        for k in raw_results.keys():
            x = raw_results[k]
            y = corrected_results[k]
            ordered_x.append(x)
            ordered_y.append(y)

        plt.scatter([m[0] for m in ordered_x],[m[0] for m in ordered_y])
        plt.xlabel('Variability including techincal factors')
        plt.ylabel('Variability excluding techincal factors')
        os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)
        plt.show()




        # raw_variance_explained = [raw_results['Lung_mean_retrained_{}'.format(s)] for s in SIZES]
        # corrected_variance_explained = [corrected_results['Lung_mean_retrained_{}'.format(s)] for s in SIZES]
        #
        # plt.plot(raw_variance_explained, label='Raw')
        # plt.plot(corrected_variance_explained, label='Corrected')
        #
        # plt.title('Lung mean retrained')
        # plt.legend(prop={'size':15})
        #
        # # plt.xticklabels(SIZES, size=10)
        # plt.xticks(range(len(SIZES)), SIZES)
        # plt.xlabel('Patch size')
        # plt.ylabel('Variance explained')
        #
        # os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)
        # plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        # plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)
        #
        # plt.show()

    @staticmethod
    def gene_ontology_analysis():
        results = pickle.load(open(GTEx_directory + '/results/TFCorrectedFeatureAssociations/gene_ontology_analysis.pickle', 'rb'))
        ontology_results = results['Lung_mean_retrained_256']
        min_pvs = []
        for res in ontology_results:
            min_pv = 1
            for term in res:

                pv = term[2]
                if pv < min_pv:
                    min_pv = pv
            min_pvs.append(min_pv)

        idx = np.argsort(min_pvs)

        sorted_results = np.array(ontology_results)[idx]


        # np.savetxt(GTEx_directory + '/results/TFCorrectedFeatureAssociations/ontology_results.pickle', np.array(sorted_results[0:4])
        # import csv
        # with open(GTEx_directory + '/plotting/NIPS/ontology_results.csv', 'w') as csvfile:
        #     writer = csv.writer(csvfile, delimiter='|')
        #     for row in sorted_results[0]:
        #         # print(row)
        #         writer.writerow(row)
        import pdb; pdb.set_trace()

        for i in range(1):
            print ('Image feature: {}'.format(idx[i]))
            print(tabulate( [x for x in sorted_results[i] if len(x[11]) < 100] ) )





    @staticmethod
    def top_genetic_associations():
        [top_pvs, top_betas] = pickle.load(open(GTEx_directory + '/results/NIPSQuestion5/top_association_results.pickle', 'rb'))

        import matplotlib as mpl
        label_size = 5
        mpl.rcParams['xtick.labelsize'] = label_size
        mpl.rcParams['ytick.labelsize'] = label_size

        f,a = plt.subplots(7,8, figsize=(10,10))
        f.suptitle('Smallest 50 pvalues')

        for (i,entry) in enumerate(top_pvs):

            pv = entry[0]
            yID, gidx = entry[1]
            g = entry[2]
            chrom, pos = entry[3]
            y = entry[4]

            a.flatten()[i].scatter(g,y, s=5)
            a.flatten()[i].set_title("p={:0.1}".format(pv))
            a.flatten()[i].set_xlabel("{}, {}".format(chrom.decode('utf-8'),pos.decode('utf-8')), size=8)
            a.flatten()[i].set_ylabel(yID)

        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
#
        os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)
        plt.savefig(GTEx_directory + '/plotting/{group}/top_pvalues.eps'.format(group=group), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/top_pvalues.png'.format(group=group), format='png', dpi=100)



        f,a = plt.subplots(7,8, figsize=(10,10))
        f.suptitle('Largest 50 betas')
        for (i,entry) in enumerate(top_betas):
            b = entry[0]
            yID, gidx = entry[1]
            g = entry[2]
            chrom, pos = entry[3]
            y = entry[4]

            a.flatten()[i].scatter(g,y, s=5)
            a.flatten()[i].set_title("b={:01.1f}".format(b))
            a.flatten()[i].set_xlabel("{}, {}".format(chrom.decode('utf-8'),pos.decode('utf-8')),size=8)
            a.flatten()[i].set_ylabel(yID)

        plt.tight_layout()
        plt.subplots_adjust(top=0.90)

        os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)
        plt.savefig(GTEx_directory + '/plotting/{group}/top_betas.eps'.format(group=group), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/top_betas.png'.format(group=group), format='png', dpi=100)





if __name__ == '__main__':
    eval(group + '().' + name + '()')
