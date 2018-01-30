# Deep Learning for biomedical image feature extraction

The software in this repository can be used to extract visual features phenotypes from high resolution images of tissues obtained from histopathology, using convolutional neural networks. This work was carried out during my time at the Wellcome Trust Sanger Institute, working with Leopold Parts and Oliver Stegle.

The purpose of this project is to understand how genetic and gene expression variation between tissue donors affect variation visual phenotypes in donor tissues. This repository is designed to be executable by anyone with access to the EBI filesystem, and permission to the directory `/hps/nobackup/research/stegle/users/willj`. We use the data that is made available as part of this Genotype Tissue Expression Project [GTEx](https://www.gtexportal.org/home/) to answer this question.

### Prerequisites

To efficiently read in, and extract tiles at different scales from the the large histopathology images (stored as SVS files), we use [OpenSlide](http://openslide.org/). Installation instructions can be found on the [GitHub repository](https://github.com/openslide/openslide).

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


To install the above packages, use:
`conda install scikit-learn keras-gpu jupyter matplotlib openslide limix seaborn opencv`

If while running a script you find that a package is missing, use `conda install <package_name>` to install it into your conda environment.

If you encounter any problems, please either leave a Github issue on this repository, or email me at williamgdjones@gmail.com. It is possible that necessary packages have been omitted.

## Authors

* - *Initial work* - [**William Jones**](https://github.com/willgdjones)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Instructions
These following steps are need to replicate the work:
1. Download the images
2. Sample tissue patches.
3. Train the classifier.
4. Generate the features (phenotypes).
5. Perform the association tests.

### 1. Downloading the images
Given the 10 tissues types contained in `textfiles/tissue_types.txt`, the command `make download` with download all available images for each tissues and save them into `data/raw`. Note that this will take a long time, so I advise to leave this running overnight.

### 2. Sampling tissue patches.
For each image of the tissue, we sample square patches of 6 different pixel sizes (128,256,512,1024,2048,4096) such that the center of the patches lie within the tissue boundary. The `make sample_patches` performs this, and saves the resulting patches into HDF5 files within `data/patches`. Note that all images need to already be downloaded for this step to complete successfully.

### 3. Training the classifier
The feature extraction method that we use relies on a fine-tuned convolutional neural network that has been trained to differentiate between different types of tissue, from square patches. To train this neural network, we compile a dataset of 5000 random image patches at a patch size of 128 from the 10 different tissue types, and run the script `src/classifier/tissue_classifier.py`, to train the neural network. This final model is saved as the file: `models/inception_50_-1.h5`. If you wish to replicate this step, please get in contact with me. Otherwise, it is sufficient to use the previously trained model.

### 4. Generating features.
Following this, I generate a feature representation for each patch that is sampled within each image (from step 2). This is done by passing every square patch within the tissue boundary of an image through the network, and extracting the final layer. In the retrained CNN model, this final layer is a length 1024 vector. These collected features are stored in the HDF5 file `data/raw/collected_features.h5py` which has the following hierarchical structure:

`/<tissue>/<layer>/<patch_size>/<model>/<ID>`

Descriptors:
- `tissue`: Specify which tissue out the of the 10 available tissue types. e.g. 'Lung'
- `layer`: Features extracted at different layers of the network. Options: [-1, 1, 7, 12, 65, 166]. e.g. -1 = last layer, 7 = 7th layer.
- `patch_size`: The patch_size used to generate the features [128,256,512,1024,2056,4096]
- `model`: raw or retrained. Raw model uses InceptioNet architecture + ImageNet weights. Retrained is after retraining on tissue classes.

For example, the following python code will extract all the final layer features from the Lung sample with ID GTEX-ZYY3-0926 at a patch size of 128.

```
import h5py
with h5py
filename = 'data/h5py/collected_features.h5py'

with h5py.File(filename) as f:
    features = f['Lung']['-1']['128']['retrained']['GTEX-ZYY3-0926']['features']
```

### 5. Aggregating Features.
In order to perform the association tests with genotype and expression data, we need to aggregate all features for a given image. These aggregations are stored in `data/h5py/aggregated_features.h5py`. This has the following hierarchical structure:

`/<tissue>/<layer>/<patch_size>/<model>/<aggregation>`

These descriptors are the same as above, `<aggregation>` takes on max, median, and mean.

At the `<layer>` level, there is also the keys:

`'donorIDs', 'genotype_locations', 'ordered_expression', 'ordered_genotypes', 'transcriptIDs'`

These contain ordered arrays of the genotype matrices and gene expression matrices with the corresponding genotype locations and transcript IDs. These matrices can be loaded to efficiently calculate the correlation between each gene or genotype and the neural network features across the available donors.

### Correcting gene expression data with known covariates

As per the methods in [Aguet, F.](Genetic effects on gene expression across human tissues. http://doi.org/10.1038/nature24277), we regress out both known technical covariates and PEER factors.
These steps are contained in the Jupyter Notebook.

`notebooks/daily/17-11-08-correct-v7-data.ipynb`


### Contributing
For future collaborators who would like to continue this project, and make sense of the associations that are present. I would recommend the reading through the notebook above, and replicating the association calculations. They do not take a long time to recompute, and are an instructive exercise. 
