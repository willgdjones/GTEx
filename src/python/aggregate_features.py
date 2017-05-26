convolutional_aggregations = ['mean', 'median', 'max']
aggregations = ['mean', 'median', 'max']

if l != -1:
    for cagg in convolutional_aggregations:
        print ('Square aggregation: {}'.format(cagg))
        num_filters = model_features.shape[-1]
        model_features = model_features.reshape(num_patches,-1,num_filters)
        model_features1 = eval('np.{}(model_features, axis=1)'.format(cagg))
        for agg in aggregations:
            print ('Vector aggregation: {}'.format(agg))

            model_features2 = eval('np.{}(model_features1,axis=0)'.format(agg))
            print ('Feature shape: {}'.format(model_features2.shape))

            feature_path = GTEx_directory + '/data/features/{tissue}/{model_choice}_{ID}_{patch_size}_l{layer}_a-{agg}_ca-{cagg}.hdf5'.format(tissue=tissue,model_ch

            with h5py.File(feature_path,'w') as h:
                h.create_dataset('features', data=model_features2)
    else:
        for agg in aggregations:
            print ('Vector aggregation: {}'.format(agg))

            model_features2 = eval('np.{}(model_features,axis=0)'.format(agg))

            print ('Feature shape: {}'.format(model_features2.shape))

            feature_path = GTEx_directory + '/data/features/{tissue}/{model_choice}_{ID}_{patch_size}_l{layer}_a-{agg}.hdf5'.format(tissue=tissue,model_choice=model_cho

            with h5py.File(feature_path,'w') as h:
                h.create_dataset('features', data=model_features2)
