"""Module with dataset class, and data pre-processing (transforms)."""

import os, glob
import numbers
import random
import pickle
import operator

import numpy as np
import torch
from PIL import Image

from lib.utils import read_csv, wrap_cuda


class RandomCrop(object):
    """Takes a random crop of a PIL.Image with given dimensions.

    Attributes:
        size (tuple/int): image dimension after cropping (height, weight)
        , if a single integer i is passed, output dimension is (i,i)

    """

    def __init__(self, size):
        """Constructor.

        Args:
            size (tuple/int): image dimension after cropping (height, weight)
            , if a single integer i is passed, output dimension is (i,i)

        """
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """Call random cropping function.

        Args:
            img (PIL.Image): image to be pre-processed

        Returns:
            list: pre-processed data-point containing image, and label if inputted

        """
        width, height = img.size
        target_height, target_width = self.size
        if height < target_height:
            img = img.resize((width, target_height), resample=Image.BICUBIC)
            width, height = img.size
            if label is not None:
                for i in range(len(label)):
                    label[i] = label[i].resize((width, target_height), resample=Image.NEAREST)
        if width < target_width:
            img = img.resize((target_width, height), resample=Image.BICUBIC)
            width, height = img.size
            if label is not None:
                for i in range(len(label)):
                    label[i] = label[i].resize((target_width, height), resample=Image.NEAREST)
        if width == target_width and height == target_height:
            return [img, label]
        crop_width = random.randint(0, width - target_width)
        crop_height = random.randint(0, height - target_height)
        cropped_img = img.crop((crop_width, crop_height, crop_width + target_width,
                                crop_height + target_height))

        return cropped_img


class IncImagesDataset:
    """Dataset for the challenge."""

    def __init__(self, data_path, mode='train', transform=None, cache_dir='cache',
                 n_trainable_subset=None, reload_labels=False, logger=None):
        """Constructor."""
        self.data_path = data_path
        self.mode = mode
        self.transform = transform
        self.cache_dir = cache_dir
        self.n_trainable_subset = n_trainable_subset
        self.logger = logger

        if mode != 'test' and mode != 'finetune':
            self.train_images, self.val_images = self.get_train_val_split()
            cache_exists = self.load_labels_from_cache()
            if not cache_exists or reload_labels:
                self.logger.info("Creating label dicts, this takes a few minutes ...")
                self.trainval_human_labels = self.read_trainval_human_labels()
                self.trainval_machine_labels = self.read_trainval_machine_labels()
                # self.trainval_machine_labels = {}
                self.trainval_bbox_labels = self.read_trainval_bbox_labels()
                self.save_labels_to_cache()

        self.test_labels = self.read_tuning_labels_stage_1()
        self.fine_tune_samples = list(self.test_labels.keys())
        self.label_map, self.reverse_label_map = self.get_label_map()
        self.n_trainable_classes = len(self.label_map)

        # If range of most frequent classes to train not specified, use all
        if self.n_trainable_subset is None:
            self.n_trainable_subset = (0, self.n_trainable_classes)

        self.classes_subset = self.get_n_most_frequent_classes(self.trainval_human_labels,
                                                               self.n_trainable_subset[0],
                                                               self.n_trainable_subset[1])

        if mode != 'test' and mode != 'finetune':
            # Remove samples which don't have a trainable class
            self.filter_set_based_on_trainable_classes(set='train')
            # self.filter_set_based_on_trainable_classes(set='val')
            self.logger.info("No. of train images: {}, val images: {}".format(len(self.train_images),
                                                                         len(self.val_images)))
            self.logger.info("No. of trainable classes: {}".format(len(self.classes_subset)))

        self.test_images = self.get_test_image_list()
        if mode == 'test':
            self.logger.info("No. of test images: {}".format(len(self.test_images)))
            self.logger.info("No. of trainable classes: {}".format(len(self.classes_subset)))

    def get_n_trainable_classes(self):
        """Return n_trainable_classes attribute.

        Returns:
            (int): number of trainable classes.

        """
        return self.n_trainable_classes

    def get_reverse_label_map(self):
        """Return reverse_label_map attribute.

        Returns:
            (dict): dict for reverse label mapping.

        """
        return self.reverse_label_map

    def load_labels_from_cache(self):
        """Load label dicts from cache."""
        human_labels_cache_path = os.path.join(self.cache_dir, 'human_labels.pkl')
        machine_labels_cache_path = os.path.join(self.cache_dir, 'machine_labels.pkl')
        bbox_labels_cache_path = os.path.join(self.cache_dir, 'bbox_labels.pkl')

        if os.path.isfile(human_labels_cache_path) and \
            os.path.isfile(machine_labels_cache_path) and \
                os.path.isfile(machine_labels_cache_path):
            self.logger.info("Loading label dicts from cache ...")
            with open(human_labels_cache_path, 'rb') as handle:
                self.trainval_human_labels = pickle.load(handle)
            with open(machine_labels_cache_path, 'rb') as handle:
                self.trainval_machine_labels = pickle.load(handle)
            with open(bbox_labels_cache_path, 'rb') as handle:
                self.trainval_bbox_labels = pickle.load(handle)
            return True
        else:
            return False

    def save_labels_to_cache(self):
        """Save label dicts to cache."""
        os.makedirs(self.cache_dir, exist_ok=True)
        human_labels_cache_path = os.path.join(self.cache_dir, 'human_labels.pkl')
        machine_labels_cache_path = os.path.join(self.cache_dir, 'machine_labels.pkl')
        bbox_labels_cache_path = os.path.join(self.cache_dir, 'bbox_labels.pkl')

        with open(human_labels_cache_path, 'wb') as handle:
            pickle.dump(self.trainval_human_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(machine_labels_cache_path, 'wb') as handle:
            pickle.dump(self.trainval_machine_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(bbox_labels_cache_path, 'wb') as handle:
            pickle.dump(self.trainval_bbox_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        """Get length of dataset.

        Returns:
            (int): size of dataset.

        """
        if self.mode == 'train':
            return len(self.train_images)
        elif self.mode == 'val':
            return len(self.val_images)
        elif self.mode == 'test':
            return len(self.test_images)
        elif self.mode == 'finetune':
            return len(self.fine_tune_samples)

    def get_train_val_split(self):
        """Split training set for competition to train/val.

        Returns:
            (tuple): tuple with list of paths to training and validation images.

        """
        with open("trainval_image_paths.txt", "r") as f:
            trainval_images = f.read().splitlines()
            random.seed(500)
            random.shuffle(trainval_images)
            len_trainval_set = len(trainval_images)
            train_images = trainval_images[0:int(1 * len_trainval_set)]
            val_images = trainval_images[int(0.95 * len_trainval_set):]

        return train_images, val_images

    def filter_set_based_on_trainable_classes(self, set='train'):
        """Remove samples if they don't contain a trainable class.

        Args:
            set (str): 'train' or 'val' based on which set to filter.

        """
        filtered_list = []
        if set == 'train':
            filter_list = self.train_images
        elif set == 'val':
            filter_list = self.val_images
        n_samples_original_set = len(filter_list)

        for image_path in filter_list:
            image_id = image_path.split('/')[-1].split('.')[0]
            label_human = self.trainval_human_labels.get(image_id,
                                                         {'labels': [None], 'confidences': [None]})
            label_machine = self.trainval_machine_labels.get(image_id,
                                                             {'labels': [None], 'confidences': [None]})
            label_bbox = self.trainval_bbox_labels.get(image_id,
                                                       {'labels': [None], 'confidences': [None]})
            label = self.merge_labels(label_human, label_machine, label_bbox)
            for lab in label['labels']:
                if lab in self.classes_subset:
                    # even if 1 label is found, add image and then exit loop
                    filtered_list.append(image_path)
                    break
        if set == 'train':
            self.train_images = filtered_list
        elif set == 'val':
            self.val_images = filtered_list

        n_samples_filtered = len(filtered_list)
        n_samples_removed = n_samples_original_set - n_samples_filtered
        self.logger.info("Size of filtered {} set: {}, {} samples removed".format(set,
                                                                             n_samples_filtered,
                                                                             n_samples_removed))

    def get_class_map_dict(self):
        """Read the class descriptions to a dict.

        Returns:
            class_map (dict): class mapping.

        """
        class_map = {}
        file_path = os.path.join(self.data_path, 'misc', 'class-descriptions.csv')
        contents = read_csv(file_path)

        for idx, item in enumerate(contents):
            if item[0][0] == '/':
                class_map[item[0]] = item[1]

        return class_map

    def read_trainable_classes(self):
        """Read file with trainable classes."""
        file_path = os.path.join(self.data_path, 'misc', 'classes-trainable.csv')
        contents = read_csv(file_path)
        class_map_dict = self.get_class_map_dict()
        trainable_classes = {}

        for item in contents:
            label_code = item[0]
            if label_code in class_map_dict:
                trainable_classes[label_code] = class_map_dict[label_code]

        return trainable_classes

    def get_label_map(self):
        """Map string labels to indices."""
        trainable_classes = self.read_trainable_classes()
        label_map, reverse_label_map = {}, {}
        index = 0

        for key, cls_name in trainable_classes.items():
            label_map[key] = index
            reverse_label_map[index] = key
            index += 1

        return label_map, reverse_label_map

    def read_trainval_human_labels(self):
        """Read the training labels annotated by humans."""
        human_train_labels = {}
        file_path = os.path.join(self.data_path, 'misc', 'train_human_labels.csv')
        contents = read_csv(file_path)
        class_map = self.get_class_map_dict()

        for item in contents:
            image_id = item[0]
            source = item[1]
            label_code = item[2]
            confidence = item[3]
            if label_code in class_map:
                if not human_train_labels.get(image_id):
                    human_train_labels[image_id] = {'labels': [], 'confidences': []}
                else:
                    human_train_labels[image_id]['labels'].append(label_code)
                    human_train_labels[image_id]['confidences'].append(confidence)

        return human_train_labels

    def read_trainval_machine_labels(self):
        """Read the training labels annotated by machines."""
        machine_train_labels = {}
        file_path = os.path.join(self.data_path, 'misc', 'train_machine_labels.csv')
        contents = read_csv(file_path)
        class_map = self.get_class_map_dict()

        for item in contents:
            image_id = item[0]
            source = item[1]
            label_code = item[2]
            confidence = item[3]
            if label_code in class_map:
                if not machine_train_labels.get(image_id):
                    machine_train_labels[image_id] = {'labels': [], 'confidences': []}
                else:
                    machine_train_labels[image_id]['labels'].append(label_code)
                    machine_train_labels[image_id]['confidences'].append(confidence)

        return machine_train_labels

    def read_trainval_bbox_labels(self):
        """Read the bbox training labels provided by open images."""
        bbox_train_labels = {}
        file_path = os.path.join(self.data_path, 'misc', 'train_bounding_boxes.csv')
        contents = read_csv(file_path)
        class_map = self.get_class_map_dict()

        for item in contents:
            image_id = item[0]
            source = item[1]
            label_code = item[2]
            confidence = item[3]
            if label_code in class_map:
                if not bbox_train_labels.get(image_id):
                    bbox_train_labels[image_id] = {'labels': [], 'confidences': []}
                else:
                    bbox_train_labels[image_id]['labels'].append(label_code)
                    bbox_train_labels[image_id]['confidences'].append(confidence)

        return bbox_train_labels

    def get_test_image_list(self):
        """Get list of paths to test images."""
        images_path = os.path.join(self.data_path, 'misc', 'stage_1_test_images')
        images = glob.glob(os.path.join(images_path, '*.jpg'))

        return images

    def read_tuning_labels_stage_1(self):
        """Read the labels proved for tuning for stage-1 (test)."""
        tuning_labels = {}

        file_path = os.path.join(self.data_path, 'misc', 'tuning_labels.csv')
        # NOTE: uncomment to use pseudo labels
        # file_path = os.path.join('pseudo_labels.csv')
        contents = read_csv(file_path)

        for idx, item in enumerate(contents):
            image_id = item[0]
            labels = item[1].split(' ')
            tuning_labels[image_id] = {'labels': labels, 'confidences': [1] * len(labels)}
        return tuning_labels

    def get_n_most_frequent_classes(self, label_set, lower, upper):
        """Get most frequent classes from the training / tuning set."""
        labels_freq = {}

        for key, item in label_set.items():
            for label in item['labels']:
                if not label in labels_freq:
                    labels_freq[label] = 1
                else:
                    labels_freq[label] += 1

        label_ids = list(labels_freq.keys())
        label_ids = list(reversed([x for _, x in sorted(zip(list(labels_freq.values()),
                                                            label_ids))]))

        trainable_label_ids = {}
        for label in label_ids[lower:upper]:
            trainable_label_ids[label] = labels_freq[label]

        return trainable_label_ids

    def get_class_weights(self):
        """Get class weights."""
        class_weights = np.zeros(self.n_trainable_classes)
        for label_id in self.classes_subset:
            cls_index = self.label_map.get(label_id, None)
            if cls_index is not None:
                class_weights[cls_index] = 1
        class_weights = np.ones(self.n_trainable_classes)
        class_weights = wrap_cuda(torch.from_numpy(class_weights).float())

        return class_weights

    def get_labelids_sorted_by_freq(self, label_dict, labels_freq=None, test=False):
        """Get the label ids based on frequency of occurence."""
        if labels_freq is None:
            labels_freq = {}
        for key, item in label_dict.items():
            if test:
                label_list = item
            else:
                label_list = item['labels']
            for label in label_list:
                if not label in labels_freq:
                    labels_freq[label] = 0
                else:
                    labels_freq[label] += 1
        label_ids = list(labels_freq.keys())
        label_freqs = list(labels_freq.values())
        label_ids = list(reversed([x for _, x in sorted(zip(label_freqs, label_ids))]))

        return label_ids

    def write_trainval_image_list(self):
        """Get list of paths to train images, write them to txt file."""
        train_image_list = []
        folder_list = ['train_0', 'train_1', 'train_2', 'train_3', 'train_4',
                       'train_5', 'train_6', 'train_7', 'train_8', 'train_9',
                       'train_a', 'train_b', 'train_c', 'train_d', 'train_e',
                       'train_f']
        for folder in folder_list:
            images_path = os.path.join(self.data_path, 'train_images', folder)
            train_image_list.extend(glob.glob(os.path.join(images_path, '*.jpg')))
        with open('trainval_image_paths.txt', 'w') as f:
            for item in train_image_list:
                f.write("%s\n" % item)

    def merge_labels(self, human_label, machine_label, bbox_label):
        """Merge list of all annotations.

        Args:
            human_label (dict): dict with human verified labels.
            machine_label (dict): dict with machine generated labels.
            bbox_label (dict): dict with labels from bbox set.

        Returns:
            merge_labels (dict): merged labels dict.

        """
        merged_label = {'labels': [], 'confidences': []}
        for idx, item in enumerate(human_label['labels']):
            merged_label['labels'].append(item)
            merged_label['confidences'].append(human_label['confidences'][idx])
        for idx, item in enumerate(bbox_label['labels']):
            if not item in merged_label['labels']:
                merged_label['labels'].append(item)
                merged_label['confidences'].append(bbox_label['confidences'][idx])
        for idx, item in enumerate(machine_label['labels']):
            if not item in merged_label['labels']:
                merged_label['labels'].append(item)
                merged_label['confidences'].append(machine_label['confidences'][idx])
        return merged_label

    def __getitem__(self, index):
        """Get one item for data loading.

        Args:
            index (int): index of sample.

        Returns:
            (tuple): tuple with image, label vector and image ID.

        """
        if self.mode == 'train':
            image_path = self.train_images[index]
        elif self.mode == 'val':
            image_path = self.val_images[index]
        elif self.mode == 'test':
            image_path = self.test_images[index]
        elif self.mode == 'finetune':
            image_id = self.fine_tune_samples[index]
            image_path = os.path.join(self.data_path, 'misc',
                                      'stage_1_test_images', image_id + '.jpg')

        img = Image.open(image_path)
        img = self.transform(*[img])
        image_id = image_path.split('/')[-1].split('.')[0]

        if self.mode == 'train' or self.mode == 'val':
            label_human = self.trainval_human_labels.get(image_id,
                                                         {'labels': [None], 'confidences': [None]})
            label_machine = self.trainval_machine_labels.get(image_id,
                                                             {'labels': [None], 'confidences': [None]})
            label_bbox = self.trainval_bbox_labels.get(image_id,
                                                       {'labels': [None], 'confidences': [None]})
            label = self.merge_labels(label_human, label_machine, label_bbox)
        else:
            label = self.test_labels.get(image_id, {'labels': [None], 'confidences': [None]})

        label_vector = self.label_to_vector(label)

        one_hot_vectors = self.labels_to_one_hot(label)

        return img, label_vector, one_hot_vectors, image_id

    def label_to_vector(self, labels):
        """Convert string of labels to binary vector.

        Args:
            labels (dict): dict of labels.

        Returns:
            label_vector (numpy.ndarray): label vector.

        """
        label_vector = np.zeros(self.n_trainable_classes)
        for idx, label in enumerate(labels['labels']):
            # mapped to trainable classes
            if label in self.classes_subset:
                label_index = self.label_map.get(label, None)
                label_confidence = labels['confidences'][idx]
                if label_index is not None:
                    label_vector[label_index] = label_confidence

        return label_vector

    def labels_to_one_hot(self, labels):
        """Take dict of labels and create one hot vectors.

        Args:
            labels (dict): dict of labels.

        Returns:
            label_vectors (list): list of one-hot vectors.

        """
        label_vectors = []
        for idx, label in enumerate(labels['labels']):
            label_vector = np.zeros(self.n_trainable_classes)
            # mapped to trainable classes
            if label in self.classes_subset:
                label_index = self.label_map.get(label, None)
                if label_index is not None:
                    label_vector[label_index] = 1
                    label_vector = np.expand_dims(label_vector, 0)
                    label_vectors.append((label_vector, self.classes_subset[label]))

        label_vectors.sort(key=operator.itemgetter(1), reverse=True)
        label_vectors = [torch.from_numpy(label_vector_tuple[0]).float() \
            for label_vector_tuple in label_vectors]

        # label of zeros when there are no labels
        if len(label_vectors) == 0:
            label_vector = np.zeros(self.n_trainable_classes)
            label_vector = np.expand_dims(label_vector, 0)
            label_vectors.append(torch.from_numpy(label_vector).float())

        return label_vectors
