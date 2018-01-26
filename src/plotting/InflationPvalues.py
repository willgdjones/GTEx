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


GTEx_directory = '.'

parser = argparse.ArgumentParser(description='Collection of plotting results. Runs on local computer.')
parser.add_argument('-g', '--group', help='Plotting group', required=True)
parser.add_argument('-n', '--name', help='Plotting name', required=True)
args = vars(parser.parse_args())
group = args['group']
name = args['name']

class InflationPvalues():

    @staticmethod
    def raw_pvalues():
        import seaborn as sns
        sns.set_style("dark")

        results = pickle.load(open(GTEx_directory + '/results/{group}/{name}.pickle'.format(group=group, name=name), 'rb'))
        Rs_real, pvs_real, pvs_1, pvs_2, pvs_3 = results

        from limix.plot import qqplot

        qqplot(pvs_real.flatten())
        qqplot(pvs_1.flatten())
        qqplot(pvs_2.flatten())
        qqplot(pvs_3.flatten())

        os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)

        plt.show()


    @staticmethod
    def raw_vs_pc_corrected_pvalues():

        print("Loading pvalues")
        raw_results = pickle.load(open(GTEx_directory + '/results/InflationPvalues/raw_pvalues.pickle', 'rb'))
        pc_corrected_results = pickle.load(open(GTEx_directory + '/results/InflationPvalues/pc_corrected_pvalues.pickle', 'rb'))

        import seaborn as sns
        sns.set_style("dark")
        from limix.plot import qqplot

        _, pvs_real_raw, _, _, _ = raw_results
        _, pvs_real_corrected, _, _, _ = pc_corrected_results[0]

        print('Estimating lambda for raw pvalues')
        raw_lamb = estimate_lambda(pvs_real_raw.flatten())
        print('Estimating lambda for corrected pvalues')
        corrected_lamb =  estimate_lambda(pvs_real_pc_corrected.flatten())

        print('Plotting raw pvalues')
        qqplot(pvs_real_raw.flatten(), label='raw $\lambda={:0.2f}$'.format(raw_lamb))
        print('Plotting corrected pvalues')

        qqplot(pvs_real_pc_corrected.flatten(), label='corrected $\lambda={:0.2f}$'.format(pc_corrected_lamb))
        plt.legend(prop={'size':15})

        os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)

        plt.show()

    @staticmethod
    def pc_corrected_pvalues():
        print ('Loading corrected pvalues')
        pc_corrected_results = pickle.load(open(GTEx_directory + '/results/InflationPvalues/pc_corrected_pvalues.pickle', 'rb'))


        import seaborn as sns
        sns.set_style("dark")
        from limix.plot import qqplot

        for (i, pvalues) in enumerate(pc_corrected_results):
            Rs_real, pvs_real, pvs_1, pvs_2, pvs_3 = pvalues
            print ('Calculating lambda for PC {}'.format(i))
            lamb = estimate_lambda(pvs_real.flatten())
            print ('Plotting PC {}'.format(i))
            qqplot(pvs_real.flatten(), label=r'{} PCs, $\lambda={:0.2f}$'.format(i+1, lamb))

        plt.legend(prop={'size':15}, loc='upper left')

        os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)

        plt.show()

    @staticmethod
    def tf_corrected_pvalues():

        print("Loading pvalues")
        raw_results = pickle.load(open(GTEx_directory + '/results/InflationPvalues/raw_pvalues.pickle', 'rb'))
        tf_corrected_results = pickle.load(open(GTEx_directory + '/results/InflationPvalues/tf_corrected_pvalues.pickle', 'rb'))


        import seaborn as sns
        sns.set_style("dark")
        from limix.plot import qqplot

        _, pvs_real_raw, _, _, _ = raw_results
        _, pvs_real_tf_corrected, _, _, _ = tf_corrected_results

        print('Estimating lambda for raw pvalues')
        raw_lamb = estimate_lambda(pvs_real_raw.flatten())
        print('Estimating lambda for TF corrected pvalues')
        tf_corrected_lamb =  estimate_lambda(pvs_real_tf_corrected.flatten())

        print('Plotting raw pvalues')
        qqplot(pvs_real_raw.flatten(), label='raw $\lambda={:0.2f}$'.format(raw_lamb))
        print('Plotting TF corrected pvalues')
        qqplot(pvs_real_tf_corrected.flatten(), label='corrected $\lambda={:0.2f}$'.format(tf_corrected_lamb))

        plt.legend(prop={'size':15})

        os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)
        plt.show()

    @staticmethod
    def tf_vs_pc_corrected_pvalues():

        print("Loading pvalues")
        pc_corrected_results = pickle.load(open(GTEx_directory + '/results/InflationPvalues/pc_corrected_pvalues.pickle', 'rb'))
        tf_corrected_results = pickle.load(open(GTEx_directory + '/results/InflationPvalues/tf_corrected_pvalues.pickle', 'rb'))


        import seaborn as sns
        sns.set_style("dark")
        from limix.plot import qqplot

        _, pvs_real_pc_corrected, _, _, _ = pc_corrected_results[0]
        _, pvs_real_tf_corrected, _, _, _ = tf_corrected_results

        print('Estimating lambda for raw pvalues')
        pc_corrected_lamb = estimate_lambda(pvs_real_pc_corrected.flatten())
        print('Estimating lambda for TF corrected pvalues')
        tf_corrected_lamb =  estimate_lambda(pvs_real_tf_corrected.flatten())

        print('Plotting 1 PC corrected pvalues')
        qqplot(pvs_real_pc_corrected.flatten(), label='1 PC Corrected $\lambda={:0.2f}$'.format(pc_corrected_lamb))
        print('Plotting TF corrected pvalues')
        qqplot(pvs_real_tf_corrected.flatten(), label='5 TF Corrected $\lambda={:0.2f}$'.format(tf_corrected_lamb))

        plt.legend(prop={'size':15})

        os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)
        plt.show()

    @staticmethod
    def all_tf_corrected_pvalues():
        import seaborn as sns
        sns.set_style("dark")
        from limix.plot import qqplot

        print("Loading pvalues")
        all_tf_corrected_results = pickle.load(open(GTEx_directory + '/results/InflationPvalues/all_tf_corrected_pvalues.pickle', 'rb'))
        tf_corrected_lamb =  estimate_lambda(all_tf_corrected_results[0].flatten())

        qqplot(all_tf_corrected_results[0].flatten())
        qqplot(all_tf_corrected_results[1].flatten())
        plt.title('$\lambda={:0.2f}'.format(tf_corrected_lamb))
        os.makedirs(GTEx_directory + '/plotting/{}'.format(group), exist_ok=True)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.eps'.format(group=group, name=name), format='eps', dpi=100)
        plt.savefig(GTEx_directory + '/plotting/{group}/{name}.png'.format(group=group, name=name), format='png', dpi=100)
        plt.show()



if __name__ == '__main__':
    eval(group + '().' + name + '()')
