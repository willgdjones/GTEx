
import os
import numpy as np
import pdb

tissue_types = ['Lung','Artery - Tibial','Heart - Left Ventricle', 'Breast - Mammary Tissue', 'Brain - Cerebellum', 'Pancreas', 'Testis', 'Liver', 'Ovary', 'Stomach']

print ("DOWNLOAD")
print ("Number of tissues raw data {}".format(len(os.listdir('data/raw'))))
print ("\n")
