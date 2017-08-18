# import sys
# sys.path.append('/hps/nobackup/research/stegle/users/willj/GTEx',
#     '/nfs/gns/homes/willj/anaconda3/envs/GTEx/lib/python3.5/site-packages')
# import numpy as np
# import matplotlib.pyplot as plt
# import h5py
# GTEx_directory = '/hps/nobackup/research/stegle/users/willj/GTEx'
# sizes = ['128', '256', '512', '1024', '2048', '4096']
# from src.utils.helpers import *
#
# with h5py.File(GTEx_directory + '/data/h5py/aggregated_features.h5py','r') as f:
#     expression = f['Lung']['ordered_expression'].value
#     size_group = f['Lung']['-1']['256']
#     raw_features = size_group['raw']['mean']['ordered_aggregated_features'].value
#     raw_features[raw_features < 0] = 0
#     retrained_features = size_group['retrained']['mean']['ordered_aggregated_features'].value
#
#
#


# Full histogram of Mean feature Lung size 256 retrained
# plt.figure(figsize=(30, 8))
# plt.imshow(retrained_features, cmap='Reds')
# plt.title("Mean feature, Lung size 256", size=20)
# plt.xlabel("Features", size=20)
# plt.ylabel("Individuals", size=20)
# plt.colorbar()
# plt.tight_layout()
# plt.savefig('figures/exploratory/heatmaps/ + \
#             heatmap_retrained_mean_feature_lung_256.eps',
#             format='eps', dpi=100)


# # Full histogram of Mean feature Lung size 256 zoom retrained
# plt.figure(figsize=(13, 10))
# plt.imshow(retrained_features[0:50,0:50], cmap='Reds')
# plt.title("Mean feature, Lung size 256", size=20)
# plt.xlabel("Features", size=20);
# plt.ylabel("Individuals", size=20)
# plt.colorbar()
# plt.tight_layout()
# plt.savefig('figures/exploratory/heatmaps/ + \
#             heatmap_retrained_mean_feature_lung_256_zoom.eps',
#             format='eps', dpi=100)
#
#
# # Full histogram of Mean feature Lung size 256 retrained
# plt.figure(figsize=(30, 5))
# plt.imshow(raw_features, cmap='Reds')
# plt.title("Mean aggregated feature. Raw Inceptionet. Lung, patch-size 256", size=20)
# plt.xlabel("Features", size=20)
# plt.ylabel("Individuals", size=20)
# plt.colorbar()
# plt.tight_layout()
# plt.savefig('figures/exploratory/heatmaps/ + \
#         heatmap_raw_mean_feature_lung_256.eps', format='eps', dpi=100)
#
#
# # Full histogram of Mean feature Lung size 256 zoom retrained
# plt.figure(figsize=(13, 10))
# plt.imshow(raw_features[0:50, 0:50], cmap='Reds')
# plt.title("Mean aggregated feature. Raw Inceptionet. Lung, patch-size 256. Zoomed", size=20)
# plt.xlabel("Features", size=20)
# plt.ylabel("Individuals", size=20)
# plt.tight_layout()
# plt.colorbar()
# plt.savefig('figures/exploratory/heatmaps/ + \
#         zoom_heatmap_raw_mean_feature_lung_256.eps', format='eps', dpi=100)
#
#
# with h5py.File(GTEx_directory + '/data/h5py/collected_features.h5py','r') as f:
#     patch_features = f['Lung']['-1']['256']['retrained']['GTEX-111FC-1126']['features'].value
#
# plt.figure(figsize=(30, 8))
# plt.imshow(np.array(patch_features).T.astype(np.float32), cmap='Reds')
# plt.title("Image feature variation across patches", size=25)
# plt.xlabel("Patches", size=20)
# plt.ylabel("Image feature",size=20)
# plt.colorbar()
# plt.tight_layout()
# plt.savefig('figures/exploratory/heatmaps/ + \
#         heatmap_image_features_across_patches.eps', format='eps', dpi=100)
#
# plt.figure(figsize=(13, 10))
# plt.imshow(np.array(patch_features).T.astype(np.float32)[0:100, 0:100], cmap='Reds')
# plt.title("Image feature variation across patches. Zoomed", size=20)
# plt.xlabel("Patches", size=20)
# plt.ylabel("Image feature", size=20)
# plt.tight_layout()
# plt.colorbar()
# plt.savefig('figures/exploratory/heatmaps/ + \
#     heatmap_image_features_across_patches_zoomed.eps', format='eps', dpi=100)
#
# plt.figure(figsize=(20, 3))
# plt.imshow(np.array(patch_features).std(axis=0).reshape(1, -1).astype(np.float32)[:, 0:100], cmap='Reds')
# plt.yticks([])
# plt.xlabel('Image features')
# plt.title('Retrained Inceptionet, Image feature standard deviation across patch-size. Lung, patch-size 256. Features 1-100.')
# plt.colorbar()
# plt.tight_layout()
# plt.savefig('figures/exploratory/heatmaps/image_feature_standard_deviation.eps', format='eps', dpi=100)
