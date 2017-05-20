import os
from glob import glob
from os.path import join as pjoin

GTEx_dir = '/hps/nobackup/research/stegle/users/willj/GTEx'

tissues = ['Lung', 'Artery - Tibial', 'Heart - Left Ventricle', 'Breast - Mammary Tissue', 'Brain - Cerebellum', 'Pancreas', 'Testis', 'Liver', 'Ovary', 'Stomach']
patch_dir = GTEx_dir + '/data/patches/'
sizes = ['128','256','512','1024','2048','4096']

def count_size(directory,size): 
    return str(len(glob(pjoin(directory, '*{}*'.format(size)))))

print ('Monitoring patches')
print ('Tissue \t\t' + '\t'.join(sizes))
print ('-'*50)
for t in tissues:
    tissue_dir = pjoin(patch_dir,t)
    size_counts = [count_size(tissue_dir, s) for s in sizes]
    print (t.split(' ')[0][0:6] + '\t\t' + '\t'.join(size_counts))


    

