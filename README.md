# Project Title

This software in this repository can be used to extract visual features from high resolution images of tissues obtained from histopathology using neural networks. The purpose of this project is to understand how genetic variants and gene expression variation effect these visual features.

## Getting Started

### Prerequisites

To efficiently read in the large histopathology image (SVS files), we use [OpenSlide](http://openslide.org/) .
Opens

```
Give examples
```

### Installing

A step by step series of examples that tell you have to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags).

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* etc







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
