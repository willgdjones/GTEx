# GTEx_directory = '/hps/nobackup/research/stegle/users/willj/GTEx'
# import sys
# sys.path.append(GTEx_directory)
# import h5py
# import gzip
# import pandas as pd
# import numpy as np
# import pickle
# import matplotlib.pyplot as plt
# from scipy.stats import pearsonr
# import pylab as PL
# from src.utils.helpers import *
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# features, expression, donorIDs, transcriptIDs, technical_factors, technical_headers, technical_idx = extract_final_layer_data('Lung', 'retrained', 'median', '256')
#
# from sklearn.decomposition import PCA
# pca = PCA(n_components=0.95)
# pca_expression = pca.fit_transform(expression)
# pca = PCA(n_components=0.95)
# pca_features = pca.fit_transform(features)
#
#
# **Expression PCs vs technical factors**
#
# from numpy import ma
# from matplotlib import cbook
# from matplotlib.colors import Normalize
#
# class MidPointNorm(Normalize):
#     def __init__(self, midpoint=0, vmin=None, vmax=None, clip=False):
#         Normalize.__init__(self,vmin, vmax, clip)
#         self.midpoint = midpoint
#
#     def __call__(self, value, clip=None):
#         if clip is None:
#             clip = self.clip
#
#         result, is_scalar = self.process_value(value)
#
#         self.autoscale_None(result)
#         vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint
#
#         if not (vmin < midpoint < vmax):
#             raise ValueError("midpoint must be between maxvalue and minvalue.")
#         elif vmin == vmax:
#             result.fill(0) # Or should it be all masked? Or 0.5?
#         elif vmin > vmax:
#             raise ValueError("maxvalue must be bigger than minvalue")
#         else:
#             vmin = float(vmin)
#             vmax = float(vmax)
#             if clip:
#                 mask = ma.getmask(result)
#                 result = ma.array(np.clip(result.filled(vmax), vmin, vmax),
#                                   mask=mask)
#
#             # ma division is very slow; we can take a shortcut
#             resdat = result.data
#
#             #First scale to -1 to 1 range, than to from 0 to 1.
#             resdat -= midpoint
#             resdat[resdat>0] /= abs(vmax - midpoint)
#             resdat[resdat<0] /= abs(vmin - midpoint)
#
#             resdat /= 2.
#             resdat += 0.5
#             result = ma.array(resdat, mask=result.mask, copy=False)
#
#         if is_scalar:
#             result = result[0]
#         return result
#
#     def inverse(self, value):
#         if not self.scaled():
#             raise ValueError("Not invertible until scaled")
#         vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint
#
#         if cbook.iterable(value):
#             val = ma.asarray(value)
#             val = 2 * (val-0.5)
#             val[val>0]  *= abs(vmax - midpoint)
#             val[val<0] *= abs(vmin - midpoint)
#             val += midpoint
#             return val
#         else:
#             val = 2 * (val - 0.5)
#             if val < 0:
#                 return  val*abs(vmin-midpoint) + midpoint
#             else:
#                 return  val*abs(vmax-midpoint) + midpoint
#
# norm = MidPointNorm(midpoint=0)
#
# plt.figure(figsize=(12,10))
# ax1 = plt.subplot2grid((5,5), (0,0),colspan=5, rowspan=2)
# ax2 = plt.subplot2grid((5,5), (2,0),colspan=5, rowspan=2)
#
#
# divider1 = make_axes_locatable(ax1)
# divider2 = make_axes_locatable(ax2)
# cax1 = divider1.append_axes('right', size='2%', pad=0.05)
# cax2 = divider2.append_axes('right', size='2%', pad=0.05)
#
# N = technical_factors.shape[1]
# M = pca_expression[technical_idx].shape[1]
# R_matrix = np.zeros(shape=(N,M))
# pv_matrix = np.zeros(shape=(N,M))
# for i in range(N):
#     for j in range(M):
#         R, pv = pearsonr(technical_factors[:,i], pca_expression[technical_idx][:,j])
#         R_matrix[i,j] = R
#         pv_matrix[i,j] = pv
#
# im1 = ax1.imshow(R_matrix,norm=norm,cmap=PL.get_cmap("coolwarm"))
# ax1.set_title("Technical factors influence Expression PC1. Pearson R values",size=20)
# ax1.set_xlabel('Expression PCs',size=20)
# ax1.set_ylabel('Technical factors',size=20)
# plt.colorbar(im1,cax=cax1, orientation='vertical')
#
#
# im2 = ax2.imshow(-np.log10(pv_matrix),cmap=PL.get_cmap("Reds"))
# ax2.set_title("Technical factors influence Expression PC1. -log10 pvalues",size=20)
# ax2.set_xlabel('Expression PCs',size=20)
# ax2.set_ylabel('Technical factors',size=20)
# plt.colorbar(im2, cax=cax2, orientation='vertical')
#
# n_PCs = pca_expression.shape[1]
# sorted_idx = np.argsort((R_matrix**2).flatten())[::-1]
# for k in range(5):
#     axk = plt.subplot2grid((5,5), (4,k),colspan=1)
#     idx = sorted_idx[k]
#     pc = idx % n_PCs
#     tf = int(idx / n_PCs)
#     axk.scatter(technical_factors[:,tf], pca_expression[technical_idx][:,pc],s=2)
#     axk.set_title('R: {:.2}, pv: {:.1}'.format(R_matrix.flatten()[idx], pv_matrix.flatten()[idx]))
#     axk.set_ylabel('Expression PC {}'.format(pc+1))
#     axk.set_xlabel(technical_headers[tf])
#
# plt.tight_layout()
#
# plt.savefig(GTEx_directory + '/figures/associations/technical_factors_vs_pca_expression.eps',format='eps', dpi=100)
#
# # **Image feature PCs vs technical factors**
#
# fig = plt.figure(figsize=(14,10))
# ax1 = plt.subplot2grid((4,5), (0,0),colspan=2, rowspan=3)
# ax2 = plt.subplot2grid((4,5), (0,3),colspan=2, rowspan=3)
#
#
# divider1 = make_axes_locatable(ax1)
# divider2 = make_axes_locatable(ax2)
# cax1 = divider1.append_axes('right', size='2%', pad=0.05)
# cax2 = divider2.append_axes('right', size='2%', pad=0.05)
#
# N = technical_factors.shape[1]
# M = pca_features[technical_idx].shape[1]
# R_matrix = np.zeros(shape=(N,M))
# pv_matrix = np.zeros(shape=(N,M))
# for i in range(N):
#     for j in range(M):
#         R, pv = pearsonr(technical_factors[:,i], pca_features[technical_idx][:,j])
#         R_matrix[i,j] = R
#         pv_matrix[i,j] = pv
#
# im1 = ax1.imshow(R_matrix, norm=norm, cmap=plt.get_cmap("coolwarm"))
# ax1.set_title("Technical factors influence Image feature PC1. Pearson R values", size=12)
# ax1.set_xlabel('Feature PCs', size=20)
# ax1.set_ylabel('Technical factors', size=20)
#
# fig.colorbar(im1,cax=cax1, orientation='vertical')
#
#
# im2 = ax2.imshow(-np.log10(pv_matrix),cmap=plt.get_cmap("Reds"))
# ax2.set_title("Technical factors influence Image feature PC1. -log10 pvalues",size=12)
# ax2.set_xlabel('Feature PCs',size=20)
# ax2.set_ylabel('Technical factors',size=20)
# fig.colorbar(im2,cax=cax2, orientation='vertical')
#
#
# n_PCs = pca_features.shape[1]
# sorted_idx = np.argsort((R_matrix**2).flatten())[::-1]
# for k in range(5):
#     axk = plt.subplot2grid((4,5), (3,k), colspan=1)
#     idx = sorted_idx[k]
#     pc = idx % n_PCs
#     tf = int(idx / n_PCs)
#     axk.scatter(technical_factors[:,tf], pca_features[technical_idx][:,pc],s=2)
#     axk.set_title('R: {:.2}, pv: {:.1}'.format(R_matrix.flatten()[idx], pv_matrix.flatten()[idx]))
#     axk.set_ylabel('Image feature PC {}'.format(pc+1))
#     axk.set_xlabel(technical_headers[tf])
# #     a[k].set_title('PC {} vs {}'.format(pc+1,filter_colnames[tf]))
#
# plt.tight_layout()
#
# plt.savefig(GTEx_directory + '/figures/associations/technical_factors_vs_pca_feature.eps',format='eps', dpi=100)
#
# # **Image feature PCs vs expression PCs **
#
# fig = plt.figure(figsize=(14,10))
# ax1 = plt.subplot2grid((5,5), (0,0),colspan=5, rowspan=2)
# ax2 = plt.subplot2grid((5,5), (2,0),colspan=5, rowspan=2)
#
#
# divider1 = make_axes_locatable(ax1)
# divider2 = make_axes_locatable(ax2)
# cax1 = divider1.append_axes('right', size='2%', pad=0.05)
# cax2 = divider2.append_axes('right', size='2%', pad=0.05)
#
# N = pca_features.shape[1]
# M = pca_expression.shape[1]
# R_matrix = np.zeros(shape=(N,M))
# pv_matrix = np.zeros(shape=(N,M))
# for i in range(N):
#     for j in range(M):
#         R, pv = pearsonr(pca_features[:,i], pca_expression[:,j])
#         R_matrix[i,j] = R
#         pv_matrix[i,j] = pv
#
#
# im1 = ax1.imshow(R_matrix, norm=norm, cmap=PL.get_cmap("coolwarm"))
# ax1.set_title("Expression PCs vs Image feature PCs. R values",size=20)
# ax1.set_xlabel('Expression PCs',size=20)
# ax1.set_ylabel('Image Feature PCs',size=20)
# fig.colorbar(im1,cax=cax1, orientation='vertical')
#
#
# im2 = ax2.imshow(-np.log10(pv_matrix),cmap=PL.get_cmap("Reds"))
# ax2.set_title("Expression PCs vs Image feature PCs. -log10 pvalues",size=20)
# ax2.set_xlabel('Expression PCs',size=20)
# ax2.set_ylabel('Image feature PCs',size=20)
# fig.colorbar(im2,cax=cax2, orientation='vertical')
#
#
# n_PCs = pca_expression.shape[1]
# sorted_idx = np.argsort((R_matrix**2).flatten())[::-1]
# for k in range(5):
#     axk = plt.subplot2grid((5,5), (4,k),colspan=1)
#     idx = sorted_idx[k]
#     Epc = idx % n_PCs
#     Ipc = int(idx / n_PCs)
#     axk.scatter(pca_features[:,Ipc], pca_expression[:,Epc],s=2)
#     axk.set_title('R: {:.2}, pv: {:.1}'.format(R_matrix.flatten()[idx], pv_matrix.flatten()[idx]))
#     axk.set_ylabel('Image feature PC {}'.format(Ipc+1))
#     axk.set_xlabel('Expression PC {}'.format(Epc+1))
#
#
# plt.tight_layout()
#
# plt.savefig(GTEx_directory + '/figures/associations/pca_expression_vs_pca_feature.eps',format='eps', dpi=100)
