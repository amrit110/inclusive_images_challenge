# Inclusive Images Challenge

Kaggle competition https://www.kaggle.com/c/inclusive-images-challenge. Solution implemented in PyTorch, (19th place).


### How to run

* Setup data directory - To run, data directory needs to be setup as follows:

```
data_dir
└─── train_images
└─── test_images
└─── labels
    │─── class-descriptions.csv
    │─── classes-trainable.csv
    │    ...
```

* `train_images` with the training data (portion of the Open Images dataset)
is only required to replicate training process.

* `test_images` - can be stage-1 or stage-2.

* To generate predictions on stage-1 or stage-2 for submission (with provided weights)


```bash
python main.py --data-path <path_to_data_dir> --mode adapt --use-ensemble true --n-models 5
```

* The model ensemble weights are loaded from `experiments`, and predictions over the test
set are generated, and the ensemble then bootstraps itself over the predictions for 2 epochs
(emperically chosen to give best results), then the generated file is used for submission.

* To train, create a `text` file with paths to training images. This is split into train/val.
The `write_trainval_image_list` method in the `IncImagesDataset` class can be used for this.

* The script only supports single model training. Single model checkpoint files are combined to
form an ensemble which is then applied to the test set.
