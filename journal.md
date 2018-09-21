# Journal

### Data

* Each image can have multiple labels
* Training data drawn mostly from North America and Western Europe

#### Training dataset

* `1,743,042` images
* `8,036,467` human annotated labels
* `15,259,187` machine annotated labels
* `7,178` unique trainable classes

#### Fine-tuning dataset (Stage 1)

* `1,000` images
* `2,386` labels
* `484` unique classes

#### Test dataset (Stage 1)

* `32,580` images

### Relevant papers

* http://cvlab.cse.msu.edu/pdfs/DoConvolutionalNeuralNetworksLearnClassHierarchy.pdf

* https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Yan_HD-CNN_Hierarchical_Deep_ICCV_2015_paper.pdf


### Tasks:

- [ ] Train a baseline multi-class classifier, by simply cropping images to
fixed dimension and predicting for all trainable classes. Use validation set, but
also the stage-1 fine-tuning set as test set.
