# Deep Learning for biomedical image feature extraction

This software in this repository can be used to extract visual features from high resolution images of tissues obtained from histopathology using neural networks. The purpose of this project is to understand how genetic variants and gene expression variation effect these visual features. This repository is designed to be executable by anyone with access to the EBI filesystem.

### Prerequisites

To efficiently read in the large histopathology image (SVS files), we use [OpenSlide](http://openslide.org/). Installation instructions can be found on the [GitHub repository](https://github.com/openslide/openslide).

I advise using [MiniConda](https://conda.io/docs/glossary.html#miniconda-glossary) for package management. Necessary packages include:
- scikit-learn
- keras-gpu
- jupyter
- matplotlib
- numpy
- openslide
- limix
- seaborn
- OpenCV

If a script uses any other package, using `conda install <package_name>`.


## Authors

* - *Initial work* - [**William Jones**](https://github.com/willgdjones)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Instructions
These steps are need to replicate the work:
1. Downloading the images
2. Sampling tissue patches.
3. Training the classifier
4. Generating the features.
5. Performing the association tests.

### 1. Downloading the images
Given the 10 tissues types contained in `textfiles/tissue_types.txt`, the command `make download` with download all available images for each tissues and save them into `data/raw`. Note that this will take a long time, so leave this running overnight.

### 2. Sampling tissue patches.
For each image of the tissue, we sample square patches of 6 different sizes (128,256,512,1024,2048,4096) that are centered within the tissue boundary. The `make sample_patches` performs this, and saves the resulting patches into HDF5 files within `data/patches`. Note that all images need to already be downloaded for this step to complete successfully.

### 3. Training the classifier
The feature extraction method that we use relies on a fine-tuned convolutional neural network. To train this neural network, we compile a dataset of 5000 random image patches at a patch size of 128 from the 10 different tissue types, and using the script `src/classifier/tissue_classifier.py`, I train the neural network. This final model is saved as the file: `models/inception_50_-1.h5`.

### 4. Generating features.
Following this, I generate a feature representation for each patch within each image. This is done by passing every square patch within the tissue boundary of an image through the network, and extracting the final layer. This final layer is a length 1024 vector. These collected features are stored in the HDF5 file `data/raw/collected_features.h5py` which has the following hierarchical structure:

`/<tissue>/<layer>/<patch_size>/<model>/<ID>`

tissue: Specify which tissue out the of the 10 available tissue types.
layer: Features extracted at different layers of the network [-1, 1, 7, 12, 65, 166]. -1 = last layer, 7 = 7th layer.
patch_size: The patch_size used to generate the features [128,256,512,1024,2056,4096]
model: raw or retrained. Raw model uses InceptioNet architecture + ImageNet weights. Retrained is after retraining on tissue classes.

For example, the following code would extracts all the final layer features from the Lung sample with ID GTEX-ZYY3-0926 at a patch size of 128.
```
import h5py
with h5py
filename = 'data/h5py/collected_features.h5py'

with h5py.File(filename) as f:
    features = f['Lung']['-1']['128']['retrained']['GTEX-ZYY3-0926']['features']
```

### Aggregating Features.
In order to perform the association tests with genotype and expression data, we need to aggregate all features for a given image. These aggregations are stored in `data/h5py/aggregated_features.h5py`. This has the following hierarchical structure:
`/<tissue>/<layer>/<patch_size>/<model>/<aggregation>`

These descriptors are the same as above, `<aggregation>` takes on max, median, and mean.

At the <layer> level, there is also the keys:

`'donorIDs', 'genotype_locations', 'ordered_expression', 'ordered_genotypes', 'transcriptIDs'`

These contain ordered arrays of the genotype matrices and gene expression matrices with the corresponding genotype locations and transcript IDs. These matrices can be loaded to efficiently calculate the pairwise associations.

### Correcting gene expression data

The PEER factors used to correct the gene expression data are contained in the Jupyter Notebook: `notebooks/daily/17-11-08-correct-v7-data.ipynb`
