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

### Problem Description

* Improve generalisation of image classifiers to distributional shifts.

* Distributional changes come from sampling images from different geographical regions.

* Stage-1 and Stage-2 will be quite different distributions so fine-tuning much to the
stage-1 will be bad.

* Stage-1 results don't mean much unless the results come from only using the training set,
and using minimal information from the finetuning set.

* Since label distributions between Stage-1 and Stage-2 will differ significantly, we can't
restrict to use any label subset based on Stage-1 label distribution.

<p float="left">
  <img src="/plots/sample_geo_distribution.png" width="640" />   
</p>

### Relevant papers

* No Classification without Representation: Assessing Geodiversity Issues in
Open Data Sets for the Developing World - https://arxiv.org/pdf/1711.08536.pdf


### Tasks:

- [x] Train a baseline multi-class classifier, by simply cropping images to
fixed dimension and predicting for most frequent classes (10). Use only training set.

* `submission_4.csv` - `0.299`

- [x] Same as above but with 100 frequent classes, also use finetune set.

* `submission_7.csv` - `0.386`

- [x] Same as above but with 484 classes (present in finetune set), use finetune set.

* `submission_10.csv` - `0.404`

- [x] Same as above but with 484 classes (present in finetune set), but use model predictions as pseudo-labels and
fine-tune further.

* `submission_11.csv` - `0.435`


- [x] Train on all allowed classes. No fine-tuning.  

* `submission_16.csv` - `0.296`
* This is the baseline general performance. Any technique applied, should be on top of this
without using much knowledge from Stage-1 finetuning set.
* Use more augmentations like random-crops, color jitters and brightness augs to improve this baseline.


- [ ] Do error analysis on the fine-tuning images, to see where the classifier is struggling.
Compute TPs, FPs and FNs, look at the images.

* It seems like precision is quite low, there are many predicted labels and hence FPs.
This could be due to the highly noisy machine generated labels used for training. So, could be
interesting to not use them and compare. Recall is weighted more, so it might also hurt the F2-score.
