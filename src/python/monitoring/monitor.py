import os
import numpy as np
import pdb

tissue_types = ['Lung','Artery - Tibial','Heart - Left Ventricle', 'Breast - Mammary Tissue', 'Brain - Cerebellum', 'Pancreas', 'Testis', 'Liver', 'Ovary', 'Stomach']

print ("SAMPLING")
print ("Tissue | Number of Images | Number of Images with covering patches | Average number of patches per image.")
for size in ['small', 'medium', 'large']:
# for size in ['large']:
    print ("\n")
    print ("{} patches".format(size))
    for tissue in os.listdir(os.path.join('data/raw')):
        number_downloaded_images = len(os.listdir(os.path.join('data/raw', tissue)))
        # if size == 'large':
            # pdb.set_trace()
        number_images_with_covering_patches = sum(np.array([len(os.listdir(os.path.join('data/processed/covering_patches/',size,tissue,ID))) for ID in os.listdir(os.path.join('data/processed/covering_patches/',size,tissue))]) > 0)
        # number_images_with_covering_patches = len( [len(os.listdir(os.path.join('data/processed/covering_patches/',size,tissue,ID))) for ID in os.listdir(os.path.join('data/processed/covering_patches/',size,tissue))] )

        if size == 'large':
            factor = 100 
        else:
            factor = 2000

        average_patches_per_image = np.mean([len(os.listdir(os.path.join('data/processed/covering_patches/',size,tissue,ID)))*factor for ID in os.listdir(os.path.join('data/processed/covering_patches/',size,tissue))])
        try:
            number_of_rep_batches = len(os.listdir(os.path.join('data/processed/assembled_representations/inception_50_-1.h5/', size,tissue)))
        except FileNotFoundError:
            number_of_rep_batches = 0
        print (tissue, '|', number_downloaded_images, '|',number_images_with_covering_patches, '|', average_patches_per_image)
print ("\n")


print ("ASSEMBLED REPRESENTATIONS for inception_50_-1.h5")
print ("Tissue | Assembled representations available")
for size in ['small','medium','large']:
# for size in ['large']:
    print ("\n")
    print ("{} patches".format(size))
    for tissue in os.listdir('data/processed/assembled_representations/inception_50_-1.h5/{}'.format(size)):
        number_images_with_covering_patches = len( [len(os.listdir(os.path.join('data/processed/covering_patches/',size,tissue,ID))) for ID in os.listdir(os.path.join('data/processed/covering_patches/',size,tissue))] )
        number_of_rep_batches = len(os.listdir(os.path.join('data/processed/assembled_representations/inception_50_-1.h5/', size,tissue)))
        print (tissue, '|', number_images_with_covering_patches, '|', number_of_rep_batches)

print ("ASSOCIATION DATA for inception_50_-1.h5")

print ("P VALUES | GRAPHS")
for size in ['small','medium','large']:
    print ("\n")
    print ("{} patches".format(size))
    print ("Number of tissues with association data {}".format(len(os.listdir('data/processed/association_data/expression/inception_50_-1.h5/{}'.format(size)))))
    for tissue in os.listdir('data/processed/association_results/expression/inception_50_-1.h5/{}'.format(size)):
        try:
            number_pvalues = len(os.listdir('data/processed/association_results/expression/inception_50_-1.h5/{}/{}/mean/pvalues/'.format(size,tissue)))
        except FileNotFoundError:
            number_pvalues = None
        try:
            number_graphs = len(os.listdir('data/processed/association_results/expression/inception_50_-1.h5/{}/{}/mean/graphs/'.format(size,tissue)))
        except FileNotFoundError:
            number_graphs = None

        print (tissue, '|', number_pvalues, '|', number_graphs)


