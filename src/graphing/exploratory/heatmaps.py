import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import h5py

GTEx_directory = '/hps/nobackup/research/stegle/users/willj/GTEx'


retrained_mean_features = {}
with h5py.File(GTEx_directory + '/small_data/new_retrained_inceptionet_aggregations.hdf5','r') as f:
	expression = f['lung']['256']['expression'].value
	for s in ['128','256','512','1024','2048','4096']:
		size_retrained_mean_features = f['lung'][s]['mean'].value
		retrained_mean_features[s] = size_retrained_mean_features

	expression_IDs = f['lung']['256']['expression_IDs'].value

raw_mean_features = {}
with h5py.File(GTEx_directory + '/small_data/new_raw_inceptionet_aggregations.hdf5','r') as f:
	for s in ['128','256','512','1024','2048','4096']:
		size_raw_mean_features = f['lung'][s]['mean'].value
		size_raw_mean_features[size_raw_mean_features < 0] = 0
		raw_mean_features[s] = size_raw_mean_features

#Full histogram of Mean feature Lung size 256 retrained
plt.figure(figsize=(30,8))
plt.imshow(retrained_mean_features['256'],cmap='Reds')
plt.title("Mean feature, Lung size 256",size=20)
plt.xlabel("Features",size=20); plt.ylabel("Individuals",size=20)
plt.colorbar()
plt.tight_layout()
plt.savefig('graphs/exploratory/heatmap_retrained_mean_feature_lung_256.eps',format='eps', dpi=200)


#Full histogram of Mean feature Lung size 256 zoom retrained
plt.figure(figsize=(10,10))
plt.imshow(retrained_mean_features['256'][0:50,0:50],cmap='Reds')
plt.title("Mean feature, Lung size 256",size=20)
plt.xlabel("Features",size=20); plt.ylabel("Individuals",size=20)
plt.colorbar()
plt.tight_layout()
plt.savefig('graphs/exploratory/heatmap_retrained_mean_feature_lung_256_zoom.eps',format='eps', dpi=200)


#Full histogram of Mean feature Lung size 256 retrained
plt.figure(figsize=(30,8))
plt.imshow(raw_mean_features['256'],cmap='Reds')
plt.title("Mean aggregated feature. Raw Inceptionet. Lung, patch-size 256",size=20)
plt.xlabel("Features",size=20); plt.ylabel("Individuals",size=20)
plt.colorbar()
plt.tight_layout()
plt.savefig('graphs/exploratory/heatmap_raw_mean_feature_lung_256.eps',format='eps', dpi=200)


#Full histogram of Mean feature Lung size 256 zoom retrained
plt.figure(figsize=(10,10))
plt.imshow(raw_mean_features['256'][0:50,0:50],cmap='Reds')
plt.title("Mean aggregated feature. Raw Inceptionet. Lung, patch-size 256. Zoomed",size=20)
plt.xlabel("Features",size=20); plt.ylabel("Individuals",size=20)
plt.colorbar()
plt.tight_layout()
plt.savefig('graphs/exploratory/heatmap_raw_mean_feature_lung_256_zoom.eps',format='eps', dpi=200)


with h5py.File(GTEx_directory + '/data/h5py/collected_features.h5py','r') as f:
	features = f['Lung']['-1']['256']['retrained']['GTEX-111FC-1126']['features'].value

plt.figure(figsize=(30,8))
plt.imshow(np.array(features).T.astype(np.float32),cmap='Reds')
plt.title("Image feature variation across patches",size=25)
plt.xlabel("Patches",size=20); plt.ylabel("Image feature",size=20)
plt.colorbar()
plt.tight_layout()
plt.savefig('graphs/exploratory/heatmap_image_features_across_patches.eps',format='eps', dpi=200)

plt.figure(figsize=(30,8))
plt.imshow(np.array(features).T.astype(np.float32)[0:100,0:100],cmap='Reds')
plt.title("Image feature variation across patches. Zoomed",size=20)
plt.xlabel("Patches",size=20); plt.ylabel("Image feature",size=20)
plt.tight_layout()
plt.colorbar()
plt.savefig('graphs/exploratory/heatmap_image_features_across_patches_zoomed.eps',format='eps', dpi=200)

plt.figure(figsize=(20,3))
plt.imshow(np.array(features).std(axis=0).reshape(1,-1).astype(np.float32)[:,0:100],cmap='Reds')
plt.yticks([])
plt.xlabel('Image features')
plt.title('Retrained Inceptionet, Image feature standard deviation across patch-size. Lung, patch-size 256. Features 1-100.')
plt.colorbar()
plt.tight_layout()
plt.savefig('graphs/exploratory/image_feature_standard_deviation.eps',format='eps', dpi=200)
