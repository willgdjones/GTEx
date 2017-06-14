import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import h5py

GTEx_directory = '/Users/fonz/Documents/Projects/GTEx'

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



# Comparing variation for each patch size
f, a = plt.subplots(1,5, figsize=(35,5))
for (i,s) in enumerate(['128','256','512','1024','2048']):
	a[i].hist(np.std(retrained_mean_features[s],axis=0),bins=100)
	a[i].set_title(s)
plt.savefig('graphs/feature_variation.png')


# Comparing variation when concatenating all features together
plt.figure()
concatenated_features = np.vstack([retrained_mean_features['128'], retrained_mean_features['256'], retrained_mean_features['512'], retrained_mean_features['1024'], retrained_mean_features['2048'], retrained_mean_features['4096']])
plt.hist(np.std(concatenated_features,axis=0),bins=100)
cutoff = min(np.std(concatenated_features[:,np.argsort(np.std(concatenated_features,axis=0))[-500:]],axis=0))
plt.plot([cutoff, cutoff], [0, 300],c='red')
plt.title("Histogram of variance from concatenated features across patch-sizes",size=11)
plt.xlabel("Variance")
plt.ylabel("Counts")
plt.tight_layout()
plt.savefig('graphs/concatenated_feature_variation.png')

# Histogram of expression means.
# Include cutoff for top 500
plt.figure()
plt.hist(np.mean(expression,axis=0),bins=100)
cutoff = min(np.mean(expression[:,np.argsort(np.mean(expression,axis=0))[-1000:]],axis=0))
plt.plot([cutoff, cutoff], [0, 4500],c='red')
plt.title("Histogram of mean gene expression")
plt.xlabel("Mean expression")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig('graphs/mean_expression_histogram.png')

# Histogram of expression standard deviation.
# Include cutoff for top 1000
plt.figure()
plt.hist(np.std(expression,axis=0),bins=100)
cutoff = min(np.std(expression[:,np.argsort(np.std(expression,axis=0))[-1000:]],axis=0))
plt.plot([cutoff, cutoff], [0, 2500],c='red')
plt.title("Histogram of gene expression standard deviation")
plt.xlabel("Expression standard devation")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig('graphs/std_expression_histogram.png')


#Full histogram of Mean feature Lung size 256 retrained
plt.figure(figsize=(30,8))
plt.imshow(retrained_mean_features['256'],cmap='Reds')
plt.title("Mean feature, Lung size 256")
plt.xlabel("Features"); plt.ylabel("Individuals")
plt.colorbar()
plt.tight_layout()
plt.savefig('graphs/heatmap_retrained_mean_feature_lung_256.png')


#Full histogram of Mean feature Lung size 256 zoom retrained
plt.figure(figsize=(10,10))
plt.imshow(retrained_mean_features['256'][0:50,0:50],cmap='Reds')
plt.title("Mean feature, Lung size 256")
plt.xlabel("Features"); plt.ylabel("Individuals")
plt.colorbar()
plt.tight_layout()
plt.savefig('graphs/heatmap_retrained_mean_feature_lung_256_zoom.png')


#Full histogram of Mean feature Lung size 256 retrained
plt.figure(figsize=(30,8))
plt.imshow(raw_mean_features['256'],cmap='Reds')
plt.title("Mean feature, Lung size 256")
plt.xlabel("Features"); plt.ylabel("Individuals")
plt.colorbar()
plt.tight_layout()
plt.savefig('graphs/heatmap_raw_mean_feature_lung_256.png')


#Full histogram of Mean feature Lung size 256 zoom retrained
plt.figure(figsize=(10,10))
plt.imshow(raw_mean_features['256'][0:50,0:50],cmap='Reds')
plt.title("Mean feature, Lung size 256")
plt.xlabel("Features"); plt.ylabel("Individuals")
plt.colorbar()
plt.tight_layout()
plt.savefig('graphs/heatmap_raw_mean_feature_lung_256_zoom.png')
