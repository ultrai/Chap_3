# Chap_3
Classification of retinal data Normal, AMD and DME


![Alt text](https://github.com/ultrai/Chap_3/blob/master/Images/image.png "Optional title")

![Alt text](https://github.com/ultrai/Chap_3/blob/master/Images/31.png)
![Alt text](https://github.com/ultrai/Chap_3/blob/master/Images/8.png)


## Dependencies
1. [Piotr's Image & Video Matlab Toolbox](https://github.com/pdollar/toolbox) 
2. [Structured Edge Detection Toolbox ](https://github.com/pdollar/edges)
3. [Pretrained GoogLeNet](https://github.com/soumith/inception.torch)
4. [Anaconda](https://www.continuum.io/downloads)

## Dataset
Prof. Sina Farsiu's team (Duke) and Pratul Srinivasan has generously made the data available [here!](http://people.duke.edu/~sf59/Srinivasan_BOE_2014_dataset.htm)

## Model preparation
1. git clone https://github.com/soumith/inception.torch
2. cp  prepare_model.lua incepion.torch/
3. cd incepion.toch
4. th prepare_model.lua

This creates inception.t7 in working directory

## Data preparation 
1_main_train: Preprocess training data. 

1_main_test: Preprocess training data.

or 

1_main_train_unprocess: Only resizing of train data.

1_main_test_unprocess: Only resizing of test data.

2_data.py: Creates labels corresponding to traina dn test data, randomizes train data, Compute mean image. and saves in a hdf5 file.

3_main_auxi: Load pretrained GoogLenet, fine-tune, decision pooling and confidence on test set.

4_ main_psuedo_error: identified indexes of representative response at each layer and saves in error folder.

5_ representative_responses: vizualization of representative responses.
