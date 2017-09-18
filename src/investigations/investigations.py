import pickle
import numpy as np
import argparse
# from sklearn.decomposition import PCA
# from sklearn.linear_model import LinearRegression
# from matplotlib.colors import Normalize
import os
import sys
sys.path.insert(0, os.getcwd())
from src.utils.helpers import *


GTEx_directory = '/hps/nobackup/research/stegle/users/willj/GTEx'

parser = argparse.ArgumentParser(description='Collection of experiments. Runs on the cluster.')
parser.add_argument('-g', '--group', help='Experiment group', required=True)
parser.add_argument('-n', '--name', help='Experiment name', required=True)
args = vars(parser.parse_args())
group = args['group']
name = args['name']

class Classifier():

    @staticmethod
    def validation_accuracy_across_patchsize():
        os.makedirs(GTEx_directory + '/results/{}'.format(group), exist_ok=True)
        validation_accuracies = []
        for k in range(1, 7):
            histories = pickle.load(open(GTEx_directory + '/models/histories_50_-{}.py'.format(k),'rb'))
            validation_accuracies.append(histories[1]['val_acc'][-1])
        np.savetxt(GTEx_directory + '/results/{group}/{name}.txt'.format(group=group, name=name), validation_accuracies)


if __name__ == '__main__':
    eval(group + '().' + name + '()')
