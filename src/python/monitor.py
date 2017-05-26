import os
from glob import glob
from os.path import join as path_join

GTEx_dir = '/hps/nobackup/research/stegle/users/willj/GTEx'

tissues = ['Lung', 'Artery - Tibial', 'Heart - Left Ventricle', 'Breast - Mammary Tissue', 'Brain - Cerebellum', 'Pancreas', 'Testis', 'Liver', 'Ovary', 'Stomach']
patch_dir = GTEx_dir + '/data/patches/'
feature_dir = GTEx_dir + '/data/features/'
raw_image_dir = GTEx_dir + '/data/raw/'
sizes = ['128','256','512','1024','2048','4096']

def count_features_by_size(directory,size): 
    return str(len(glob(path_join(directory, '*_{}_*'.format(size)))))

def count_patches_by_size(directory,size): 
    return str(len(glob(path_join(directory, '*_{}.*'.format(size)))))

print ('Monitoring patches')
print ('Tissue \t\t' +'Exp\t' + '\t'.join(sizes))
print ('-'*50)
for t in tissues:
    tissue_patch_dir = path_join(patch_dir,t)
    tissue_raw_image_dir = path_join(raw_image_dir,t)
    exp_counts = len(os.listdir(tissue_raw_image_dir))
    size_counts = [count_patches_by_size(tissue_patch_dir, s) for s in sizes]
    print (t.split(' ')[0][0:6] + '\t\t' + str(exp_counts) + '\t' + '\t'.join(size_counts))

print ('Monitoring features')
print ('Tissue \t\t' +'Exp\t' + '\t'.join(sizes))
print ('-'*50)
for t in tissues:
    tissue_patch_dir = path_join(patch_dir,t)
    tissue_raw_image_dir = path_join(raw_image_dir,t)
    exp_counts = len(os.listdir(tissue_raw_image_dir))
    tissue_feature_dir = path_join(feature_dir,t)
    feature_size_counts = [count_features_by_size(tissue_feature_dir, s) for s in sizes]
    exp_feature_counts = 2*(exp_counts * 5 * 2 + exp_counts)
    print (t.split(' ')[0][0:6] + '\t\t' + str(exp_feature_counts) + '\t' + '\t'.join(feature_size_counts))


    

