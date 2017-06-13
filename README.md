# Deep Learning for biomedical image feature extraction.

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
I have digital images from 10 tissues types.

Number of digital images per tissue

| Tables        | Are           | Cool  |
| ------------- |:-------------:| -----:|
| col 3 is      | right-aligned | $1600 |
| col 2 is      | centered      |   $12 |
| zebra stripes | are neat      |    $1 |

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
