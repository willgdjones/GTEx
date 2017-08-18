import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np
import h5py
from openslide import open_slide

# GTEx_directory = '/hps/nobackup/research/stegle/users/willj/GTEx'
# with h5py.File(os.path.join(GTEx_directory, 'data/h5py/collected_features.h5py'), 'r') as f:
#     IDlist = list(f['Lung']['-1']['256']['retrained'])
#     features1 = f['Lung']['-1']['256']['retrained']['GTEX-117YW-0526']['features'].value
#     features2 = f['Lung']['-1']['256']['retrained']['GTEX-117YX-1326']['features'].value
#
#
# fig = plt.figure(figsize=(30, 12))
# ax11 = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=1)
# ax12 = plt.subplot2grid((2, 3), (0, 2), colspan=1, rowspan=1)
# ax21 = plt.subplot2grid((2, 3), (1, 0), colspan=2, rowspan=1)
# ax22 = plt.subplot2grid((2, 3), (1, 2), colspan=1, rowspan=1)
#
# divider1 = make_axes_locatable(ax11)
# divider2 = make_axes_locatable(ax21)
# cax1 = divider1.append_axes('right', size='2%', pad=0.05)
# cax2 = divider2.append_axes('right', size='2%', pad=0.05)
#
# ax11.set_title("Features across patches. Lung GTEX-117YW-0526")
# im1 = ax11.imshow(features1.T.astype(np.float32), cmap='Reds')
# ax11.set_xlabel('Patch', size=15)
# ax11.set_ylabel('Image feature', size=15)
# fig.colorbar(im1, cax=cax1, orientation='vertical')
# slide1 = open_slide(GTEx_directory + '/data/raw/Lung/GTEX-117YW-0526.svs')
# image1 = slide1.get_thumbnail(size=(500, 500))
# ax12.set_title("Lung GTEX-117YW-0526 image")
# ax12.imshow(image1)
# ax12.axis('off')
#
#
# ax21.set_title("Features across patches. Lung GTEX-117YX-1326")
# im2 = ax21.imshow(features2.T.astype(np.float32), cmap='Reds')
# ax21.set_xlabel('Patch', size=15)
# ax21.set_ylabel('Image feature', size=15)
# fig.colorbar(im2, cax=cax2, orientation='vertical')
# slide1 = open_slide(GTEx_directory + '/data/raw/Lung/GTEX-117YX-1326.svs')
# image1 = slide1.get_thumbnail(size=(500, 500))
# ax22.set_title("Lung GTEX-117YX-1326 image")
# ax22.imshow(image1)
# ax22.axis('off')
# plt.show()
#
# plt.savefig('figures/exploratory/patch_features_exploration.eps', format='eps', dpi=100)
