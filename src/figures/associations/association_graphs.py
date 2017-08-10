import statsmodels.stats.multitest as smm
import pickle
import matplotlib.pyplot as plt
import seaborn
import numpy as np

alphas = [0.01,0.0001,0.000001]
sizes = [128,256,512,1024,2048,4096]
aggregations = ['mean','median']

GTEx_directory = '/hps/nobackup/research/stegle/users/willj/GTEx'
[most_expressed_transcript_idx, most_varying_feature_idx, retrained_results] = pickle.load(open(GTEx_directory + '/small_data/retrained_pvalues.py','rb'))
[most_expressed_transcript_idx, most_varying_feature_idx, raw_results] = pickle.load(open(GTEx_directory + '/small_data/raw_pvalues.py','rb'))

print ("Plotting p-values across patch-size")
# Significant p-value across patch-size
plt.figure(figsize=(10,7))
plt.title("Number of significant p-values (Bonf), varying patch size and FDR", size=17)
plt.xticks(range(len(sizes)),sizes,s ize=15)
plt.xlabel('Patch size', size=15)
plt.ylabel('Number of significant pvalues (Bonf)', size=15)
colours = ['blue','red','green']
for a in aggregations[0:1]:
    for (k, alph) in enumerate(alphas):
        points = [sum(smm.multipletests(retrained_results['{}_{}_{}'.format(a,s,'pvalues')].flatten(),method='bonferroni',alpha=alph)[0]) for s in sizes]
        plt.plot(points, c=colours[k],label=alph)
plt.legend()
plt.savefig(GTEx_directory + '/figures/associations/sign_pvalues_vary_patchsize.eps',format='eps', dpi=600)


print ("Plotting p-values raw vs retrained")
# Significant p-values comparing Raw vs Retrained Inceptionet
plt.figure()
sizes = [128,256,512,1024,2048,4096]
plt.title("Number of significant p-values (Bonf 1e-6), varying patch size. Raw vs Retrained Inceptionet")
plt.plot([sum(smm.multipletests(retrained_results['{}_{}_{}'.format('mean',s,'pvalues')].flatten(),method='bonferroni',alpha=1e-6)[0]) for s in sizes],c='blue',alpha=1,label='retrained')
plt.plot([sum(smm.multipletests(raw_results['{}_{}_{}'.format('mean',s,'pvalues')].flatten(),method='bonferroni',alpha=1e-6)[0]) for s in sizes],c='red', alpha=1,label='raw')
plt.xlabel('Patch size',size=15)
plt.ylabel('Number of significant pvalues (Bonf)',size=15)
plt.xticks(range(len(sizes)),sizes,size=15)
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig(GTEx_directory + '/figures/associations/sign_pvalues_raw_vs_retrained.eps',format='eps', dpi=600)


print ("Plotting p-values mean vs median")
# Significant p-values comparing mean / median aggregation
plt.figure(figsize=(10,10))
plt.title("Comparing Aggregation methods. Median: red, Mean: blue",size=20)
plt.xticks(range(len(sizes)),alphas,size=15)
plt.xlabel('FDR',size=20)
plt.ylabel('Number of BH significant pvalues',size=20)
for a in aggregations:
    for (k,s) in enumerate([256]):
        if a == 'mean':
            c = 'red'
        else:
            c = 'blue'

        assoc_key = '{}_{}_{}'.format(a,s,'pvalues')
        print (assoc_key)
        associations = retrained_results[assoc_key]

        points = [sum(smm.multipletests(associations.flatten(),method='bonferroni',alpha=a)[0]) for a in alphas]
        plt.plot(points, c=c,label=s,alpha=1)
plt.tight_layout()
plt.savefig(GTEx_directory + '/figures/associations/sign_pvalues_mean_vs_median.eps',format='eps', dpi=600)


print ("Plotting number of features with significant transcripts")
# Number of features with significant transcripts
plt.figure(figsize=(10,10))
plt.title("Number of features with significant pvalues (Bonf), varying patch size and FDR", size=20)
plt.xticks(range(len(sizes)),sizes,size=15)
plt.xlabel('Patch size',size=15)
plt.ylabel('Number of features with significant pvalues (Bonf)',size=15)
colours = ['blue','red','green']
for a in aggregations[0:1]:
    for (k, alph) in enumerate(alphas):
        points = [sum(np.sum(smm.multipletests(retrained_results['{}_{}_{}'.format(a,s,'pvalues')].flatten(),method='bonferroni',alpha=alph)[0].reshape(retrained_results['{}_{}_{}'.format(a,s,'R')].shape),axis=1) > 0) for s in sizes]
        plt.plot(points, c=colours[k],label=alph)
plt.legend()
plt.tight_layout()
plt.savefig(GTEx_directory + '/figures/associations/features_with_sign_transcripts.eps',format='eps', dpi=600)

print ("Plotting number of transcripts with significant features")
# Number of transcripts with significant features
plt.figure(figsize=(10,10))
plt.title("Number of transcripts significant to at least 1 feature (Bonf), varying patch size and FDR", size=20)
plt.xticks(range(len(sizes)),sizes,size=15)
plt.xlabel('Patch size',size=15)
plt.ylabel('Number of transcripts significant to at least 1 feature (Bonf)',size=15)
colours = ['blue','red','green']
for a in aggregations[0:1]:
    for (k, alph) in enumerate(alphas):
        points = [sum(np.sum(smm.multipletests(retrained_results['{}_{}_{}'.format(a,s,'pvalues')].flatten(),method='bonferroni',alpha=alph)[0].reshape(retrained_results['{}_{}_{}'.format(a,s,'R')].shape),axis=0) > 0) for s in sizes]
        plt.plot(points, c=colours[k],label=alph)
plt.legend()
plt.tight_layout()
plt.savefig(GTEx_directory + '/figures/associations/transcripts_with_sign_features.eps',format='eps', dpi=600)
