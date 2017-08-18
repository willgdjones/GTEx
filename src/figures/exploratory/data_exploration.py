import os

import numpy as np
import matplotlib.pyplot as plt



GTEx_directory = '/hps/nobackup/research/stegle/users/willj/GTEx'


# ID = 'GTEX-13FH7-1726'
# tissue = 'Lung'
# patchsize = 512
#
# image_filepath = os.path.join(GTEx_directory, 'data', 'raw', tissue, ID + '.svs')
#
# image_slide = open_slide(image_filepath)
# toplevel = image_slide.level_count - 1
# topdim = image_slide.level_dimensions[-1]
# topdownsample = image_slide.level_downsamples[-1]
# topdownsampleint = int(topdownsample)
#
# toplevelslide = image_slide.read_region((0, 0), toplevel, topdim)
# toplevelslide = np.array(toplevelslide)
# toplevelslide = toplevelslide[:, :, 0:3]
# slide = toplevelslide
#
# blurredslide = cv2.GaussianBlur(slide, (51, 51), 0)
# blurredslide = cv2.cvtColor(blurredslide, cv2.COLOR_BGR2GRAY)
# T_otsu = mahotas.otsu(blurredslide)
#
# mask = np.zeros_like(slide)
# mask = mask[:, :, 0]
# mask[blurredslide < T_otsu] = 255
#
#
# downsampledpatchsize = patchsize / topdownsampleint
# xlimit = int(topdim[1] / downsampledpatchsize)
# ylimit = int(topdim[0] / downsampledpatchsize)
#
#
# # Find downsampled coords
# coords = []
# for i in range(xlimit):
#     for j in range(ylimit):
#         x = int(downsampledpatchsize/2 + i*downsampledpatchsize)
#         y = int(downsampledpatchsize/2 + j*downsampledpatchsize)
#         coords.append((x, y))
#
# # Find coords in downsampled mask
# mask_coords = []
# for c in coords:
#     x = c[0]
#     y = c[1]
#     if mask[x, y] > 0:
#         mask_coords.append(c)
#
# slidemarkings = slide.copy()
# for c in mask_coords:
#     x = c[0]
#     y = c[1]
#     slidemarkings[x-3:x+3, y-3:y+3] = [0, 0, 255]


# Show example of Otsu thresholding
# plt.figure(figsize=(15,15))
# plt.title('Tissue boundary defined by Gaussian blurring followed by Otsu thresholding',size=20)
# plt.axis('off')
# plt.imshow(cv2.bitwise_and(slide,slide,mask=mask))
# plt.savefig('figures/exploratory/tissue_mask.eps', format='eps', dpi=100)

# Show patch centers inside tissue contour
# plt.figure(figsize=(15, 15))
# plt.axis('off')
# plt.title('Blue dots represent patch centers that fit inside tissue boundary. Patchsize 256', size=20)
# plt.imshow(slidemarkings)
# plt.savefig(
#     'figures/exploratory/show_patches_inside_mask.eps', format='eps', dpi=100)
