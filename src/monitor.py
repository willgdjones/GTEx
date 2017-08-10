import os
from glob import glob
from os.path import join as path_join
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('-t', '--tissue',
                    help='Tissue', default='Lung', required=True)
parser.add_argument('-s', '--patch_size',
                    help='Patch size', default='256', required=True)
args = vars(parser.parse_args())
tissue_choice = args['tissue']
patchsize_choice = args['patch_size']

GTEx_dir = '/hps/nobackup/research/stegle/users/willj/GTEx'

tissues = ['Lung', 'Artery - Tibial', 'Heart - Left Ventricle', 'Breast - Mammary Tissue', 'Brain - Cerebellum', 'Pancreas', 'Testis', 'Liver', 'Ovary', 'Stomach']
patch_dir = GTEx_dir + '/data/patches/'
feature_dir = GTEx_dir + '/data/features/'
raw_image_dir = GTEx_dir + '/data/raw/'
sizes = ['128', '256', '512', '1024', '2048', '4096']


def count_features_by_size(directory, size):
    return str(len(glob(path_join(directory, '*_{}_*'.format(size)))))


def count_patches_by_size(directory, size):
    return str(len(glob(path_join(directory, '*_{}.*'.format(size)))))

def count_by_layer(tissue, size):
    layers = [1,7,12,65,166]
    raw_counts_re = '/data/features/{tissue}/raw*_{size}_*l{layer}_*'
    raw_counts = [len(glob(GTEx_dir + raw_counts_re.format(tissue=tissue,size=size,layer=layer))) for layer in layers]
    raw_counts.append(len(glob(GTEx_dir + '/data/features/{tissue}/raw*_{size}_*l-1.*'.format(tissue=tissue,size=size))))

    retrained_counts_re = '/data/features/{tissue}/retrained*_{size}_*l{layer}_*'
    retrained_counts_re = retrained_counts_re
    retrained_counts = [len(glob(GTEx_dir+ retrained_counts_re.format(tissue=tissue,size=size,layer=layer))) for layer in layers]
    retrained_counts.append(len(glob(GTEx_dir + '/data/features/{tissue}/retrained*_{size}_*l-1.*'.format(tissue=tissue,size=size))))
    print ('Layer \t\t' + '\t'.join([str(x) for x in layers]) + '\t' + '-1')
    print ('Raw' + '\t\t' + '\t'.join([str(x) for x in raw_counts]))
    print ('Retrain' + '\t\t' + '\t'.join([str(x) for x in raw_counts]))


print ('Monitoring patches')
print ('Tissue \t\t' +'Exp\t' + '\t'.join(sizes))
print ('-'*50)
for t in tissues:
    tissue_patch_dir = path_join(patch_dir,t)
    tissue_raw_image_dir = path_join(raw_image_dir,t)
    exp_counts = len(os.listdir(tissue_raw_image_dir))
    size_counts = [count_patches_by_size(tissue_patch_dir, s) for s in sizes]
    print (t.split(' ')[0][0:6] + '\t\t' + str(exp_counts) + '\t' + '\t'.join(size_counts))
print ('\n\n')

print('Monitoring features')
print('Tissue \t\t' +'Exp\t' + '\t'.join(sizes))
print('-'*50)
for t in tissues:
    tissue_patch_dir = path_join(patch_dir, t)
    tissue_raw_image_dir = path_join(raw_image_dir, t)
    exp_counts = len(os.listdir(tissue_raw_image_dir))
    tissue_feature_dir = path_join(feature_dir, t)
    feature_size_counts = [count_features_by_size(tissue_feature_dir, s)
                           for s in sizes]
    exp_feature_counts = 2*(exp_counts * 5 * 2 + exp_counts)
    print(t.split(' ')[0][0:6] +
          '\t\t' +
          str(exp_feature_counts) +
          '\t' +
          '\t'.join(feature_size_counts)
          )
print('\n\n')


print('Monitoring {}, patch_size {}'.format(tissue_choice, patchsize_choice))
# print ('Tissue \t\t' +'Exp\t' + '\t'.join(sizes))
print('-'*50)
count_by_layer(tissue_choice, patchsize_choice)
print('\n\n')
