import glob
import h5py
import os
import numpy as np

GTEx_dir = '/hps/nobackup/research/stegle/users/willj/GTEx'
jn = os.path.join

tissues = ['Artery - Tibial', 'Brain - Cerebellum', 'Breast - Mammary Tissue', 'Heart - Left Ventricle', 'Liver', 'Lung', 'Ovary', 'Pancreas', 'Stomach', 'Testis']
layers = ['-1', '1', '12', '166', '65', '7']
sizes = ['128','256','512','1024','2048','4096']
models = ['raw','retrained']
convolutional_aggregations = ['mean','max']
aggregations = ['mean','median','max']



def get_donor_IDs(IDlist):
	return [str(x).split('-')[1] for x in IDlist]

def get_expression_data(tissue):

	if tissue == 'Artery - Tibial':
		tissue_filename = 'Artery_Tibial'
	elif tissue == 'Heart - Left Ventricle':
		tissue_filename = 'Heart_Left_Ventricle'
	elif tissue == 'Breast - Mammary Tissue':
		tissue_filename = 'Breast_Mammary_Tissue'
	elif tissue == 'Brain - Cerebellum':
		tissue_filename = 'Brain_Cerebellum'
	else:
		tissue_filename = tissue

	tissue_expression_filepath = '/nfs/research2/stegle/stegle_secure/GTEx/download/49139/PhenoGenotypeFiles/RootStudyConsentSet_phs000424.GTEx.v6.p1.c1.GRU/ExpressionFiles/phe000006.v2.GTEx_RNAseq.expression-data-matrixfmt.c1/parse_data/44_tissues/GTEx_Data_20150112_RNAseq_RNASeQCv1.1.8_gene_rpkm_{}_normalised_without_inverse_gene_expression.txt'.format(tissue_filename)

	with open(tissue_expression_filepath, 'r') as f:
		expression_table = np.array([x.split('\t') for x in f.read().splitlines()])
		transcriptIDs = expression_table[:,0][1:]
		expression_matrix = expression_table[1:,1:].astype(np.float32)
		tissue_expression_donorIDs = [x.split('-')[1] for x in expression_table[0,:][1:]]
	return expression_table, transcriptIDs, expression_matrix, tissue_expression_donorIDs


print ('Loading Genotype data')
genotypes_filepath = '/nfs/research2/stegle/stegle_secure/GTEx/download/49139/PhenoGenotypeFiles/RootStudyConsentSet_phs000424.GTEx.v6.p1.c1.GRU/GenotypeFiles/phg000520.v2.GTEx_MidPoint_Imputation.genotype-calls-vcf.c1/parse_data/GTEx_Analysis_20150112_OMNI_2.5M_5M_450Indiv_chr1to22_genot_imput_info04_maf01_HWEp1E6_ConstrVarIDs_all_chrom_filered_maf_subset_individuals_44_tissues.hdf5'
with h5py.File(genotypes_filepath, 'r') as f:
	genotype_IDs = f['genotype']['row_header']['sample_ID'].value
	genotype_donorIDs = get_donor_IDs(genotype_IDs)
	chrom = (list(f['genotype']['col_header']['chrom'].value))
	pos = (list(f['genotype']['col_header']['pos'].value))
	genotype_locations = [chrom, pos]
	genotype_matrix = np.array(f['genotype']['matrix'].value)

def get_donorID_intersection(t,hdf5file):
	GTExIDs = []
	for s in sizes:
		GTExIDs.append(list(hdf5file[t]['1']['max'][s]['raw'].keys()))
	GTExID_intersection = list(set.intersection(*[set(x) for x in GTExIDs]))
	donorID_intersection = get_donor_IDs(GTExID_intersection)
	return donorID_intersection

def intersect_expressionIDs(tissue,donorIDs):
	_, _, _, tissue_expression_donorIDs = get_expression_data(tissue)
	donorID_intersection = set(donorIDs).intersection(tissue_expression_donorIDs)
	return list(donorID_intersection)

def intersect_genotypeIDs(donorIDs):
	donorID_intersection = set(donorIDs).intersection(genotype_donorIDs)
	return list(donorID_intersection)

def get_ordered_expression(tissue, donorIDs):
	ordered_expression = []
	expression_table, transcriptIDs, expression_matrix, tissue_expression_donorIDs = get_expression_data(tissue)

	for dID in donorIDs:
		donorID_idx = tissue_expression_donorIDs.index(dID)
		expression_row = expression_matrix[:,donorID_idx]
		ordered_expression.append(expression_row)

	ordered_expression = np.array(ordered_expression)
	return ordered_expression, transcriptIDs

def get_ordered_genotypes(donorIDs):
	ordered_genotypes = []
	for dID in donorIDs:
		donorID_idx = genotype_donorIDs.index(dID)
		genotype_row = genotype_matrix[donorID_idx,:]
		ordered_genotypes.append(genotype_row)
	ordered_genotypes = np.array(ordered_genotypes)

	return ordered_genotypes, genotype_locations



def get_ordered_features(IDs_group, donorIDs,aggregation):

	featureGTExIDs = list(IDs_group.keys())
	featureGTExIDs = np.array(featureGTExIDs)
	donorIDs = [x.decode('utf-8') for x in donorIDs]

	featuredonorIDs = get_donor_IDs(featureGTExIDs)
	intersection_idx = np.array([str(x) in donorIDs for x in featuredonorIDs])
	GTexID_intersection = featureGTExIDs[intersection_idx]

	ordered_aggregated_features = []
#     print (GTexID_intersection)
	for gID in GTexID_intersection:
		features = IDs_group[gID]['features'].value
		if len(features.shape) == 2:

			aggregated_features = eval("np.{}(features,axis=0)".format(aggregation))
		else:
			aggregated_features = features

		ordered_aggregated_features.append(aggregated_features)
	ordered_aggregated_features = np.array(ordered_aggregated_features)

	try:
		assert len(ordered_aggregated_features.shape) == 2
	except AssertionError:
		import pdb; pdb.set_trace()

	return ordered_aggregated_features


# Pseudocode
# Open file
#   for each tissue
#       donorIDs = get_donor_ID_intersection()
#       donorIDs = intersect_genotype_IDs(donorIDs)
#       donorIDs = intersect_expression_IDs(donorIDs)

#       ordere, genotype_matrix = get_ordered_genotypes(donorIDs)
#       expression_IDs, expression_matrix = get_ordered_expression()
#       save_ordered_features(t,donor_IDs, )
#       get expression data + Id



sizes = ['128','256','512','1024','2048','4096']
with h5py.File(jn(GTEx_dir,'data/h5py/collected_features.h5py'),'r') as f, h5py.File(jn(GTEx_dir,'data/h5py/aggregated_features.h5py'),'w') as g:
	for t in tissues:
		print ('Computing intersections for {}'.format(t))
		donorIDs = get_donorID_intersection(t,f)
		donorIDs = intersect_genotypeIDs(donorIDs)
		donorIDs = intersect_expressionIDs(t,donorIDs)

		print ('{} samples with image features for all scales, along with expression and genotypes'.format(len(donorIDs)))

		print ('Extracting ordered genotypes for for {}'.format(t))
		ordered_genotypes, genotype_locations = get_ordered_genotypes(donorIDs)

		print ('Extracting ordered expression for for {}'.format(t))
		ordered_expression, transcriptIDs = get_ordered_expression(t, donorIDs)

		print ('Saving datasets for {}'.format(t))
		tissue_group = g.create_group('/' + t)
		donorIDs = [x.encode('utf-8') for x in donorIDs]
		transcriptIDs = [x.encode('utf-8') for x in transcriptIDs]
		data = ['donorIDs','ordered_genotypes','genotype_locations','ordered_expression','transcriptIDs']
		for d in data:
			tissue_group.create_dataset(d,data=eval(d))

		print ('Computing ordered image features for {}'.format(t))

		for l in layers:
			if l != '-1':
				for ca in convolutional_aggregations:
					for s in sizes:
						for m in models:
							print (l,ca,s,m)
							IDs_group = f[t][l][ca][s][m]
							for a in aggregations:
								try:
									aggregated_feature_group = g.create_group('/' + '/'.join([t,l,ca,s,m,a]))
									print ('Computing ordered aggregated features for {} {} {} {} {} {}'.format(t,l,ca,s,m,a))
									ordered_aggregated_features = get_ordered_features(IDs_group, donorIDs,a)
									aggregated_feature_group.create_dataset('ordered_aggregated_features',data=ordered_aggregated_features)
								except Exception as e:
									import pdb; pdb.set_trace()



			elif l == '-1':

				for s in sizes:
					for m in models:
						print (l,s,m)
						IDs_group = f[t][l][s][m]
						for a in aggregations:
							try:
								aggregated_feature_group = g.create_group('/' + '/'.join([t,l,s,m,a]))
								print ('Computing ordered aggregated features for {} {} {} {} {}'.format(t,l,s,m,a))
								ordered_aggregated_features = get_ordered_features(IDs_group, donorIDs,a)
								aggregated_feature_group.create_dataset('ordered_aggregated_features',data=ordered_aggregated_features)
							except Exception as e:
								import pdb; pdb.set_trace()
