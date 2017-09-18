<!-- # Deep Learning for biomedical image feature extraction.

This directory contains the files and folders for my PhD project.

## Aim
 I am aiming to extract features from images using neural networks and  associate these the genetic information.

 This repository is designed to be executable by anyone at the EBI, or who uses LSF at their research institute.

## Instructions
To replicate this research, you must follow three steps:
- Download the data
- Train the classifier
- Generate the features
- Perform the association tests

## Downloading the data
I have digital images from 10 tissues types. To download them, type:
`make download`


## Generating features
`make generate_features`

## Investigating generated features
Number of digital images per tissue, vs patch size.


|Tissue     |     Exp  |   128  |   256  |   512  |   1024  |  2048  |  4096
|-----------|----------|--------|--------|--------|---------|--------|-------   
|Lung       |     341  |   338  |   338  |   338  |   338   |  338   |  338
|Artery     |     355  |   354  |   354  |   354  |   354   |  354   |  354
|Heart      |     247  |   247  |   247  |   247  |   247   |  247   |  247
|Breast     |     218  |   217  |   217  |   217  |   217   |  217   |  217
|Brain      |     137  |   137  |   136  |   136  |   136   |  136   |  136
|Pancre     |     193  |   192  |   192  |   192  |   192   |  192   |  192
|Testis     |     197  |   196  |   196  |   196  |   196   |  196   |  196
|Liver      |     138  |   137  |   137  |   137  |   137   |  137   |  137
|Ovary      |     107  |   106  |   106  |   106  |   106   |  106   |  106
|Stomac     |     201  |   200  |   200  |   200  |   200   |  200   |  200

Number of features per tissue, vs patch size.

Tissue      |    Exp   |  128   |  256   |  512   |  1024  |  2048  |  4096
------------|----------|--------|--------|--------|--------|--------|------
Lung        |    7502  |  7084  |  7084  |  7084  |  7084  |  7084  |  7084
Artery      |    7810  |  7788  |  7788  |  7788  |  7788  |  7788  |  6974
Heart       |    5434  |  4818  |  4796  |  4796  |  4796  |  4796  |  4796
Breast      |    4796  |  4620  |  4620  |  4620  |  4620  |  4620  |  4576
Brain       |    3014  |  2805  |  2794  |  2794  |  2794  |  2794  |  2794
Pancre      |    4246  |  3586  |  3586  |  3586  |  3586  |  3586  |  3586
Testis      |    4334  |  4026  |  4026  |  4026  |  4026  |  4026  |  4026
Liver       |    3036  |  2497  |  2464  |  2464  |  2464  |  2464  |  2464
Ovary       |    2354  |  2079  |  2046  |  2046  |  2046  |  2046  |  2046
Stomac      |    4422  |  3740  |  3718  |  3718  |  3718  |  3718  |  3718

For both raw and retrained InceptioNet, we extract features and 6 different layers of the network.
Layer 1-5 of the network are convolutional layers, with the following shapes.

Feature sizes
------------|------------|--------|--------|------------|--------|--------
Layer       |     1      |      7       |       12     |      65     |     166   
Shape       | (149,149,32) | (71,71,64) | (35,35, 192) | (17,17,384) | (8,8, 320)

For the two different models, the shape of the final Dense layer is different.
--------|    -1    |
Raw     |   1024   |
Retrain |   2048   |

We aggregate across the convolutional layer squares for each filter. We compare the mean and median as aggregation methods. We define this as the convolutional aggregation. For example, the convolutional aggregation of the layer 1 feature with shape (149,149,32) has shape (1,32).


For example, consider the Lung features at patch_size 256

Monitoring Lung, patch_size 256
------------|----------|--------|--------|-------|--------|--------
Layer       |    1     |  7     |  12    |  65   |   166  |   -1
Raw         |    644   |  644   |  644   |  644  |   644  |   322
Retrain     |    644   |  644   |  644   |  644  |   644  |   322 -->
