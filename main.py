"""Main script for inclusive images challenge."""

import argparse, os, glob, random, collections, copy
import numbers
import pandas as pd
import logging
from PIL import Image
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from utils import read_csv, setup_logging, save_checkpoint, compute_f_score
from utils import custom_collate

from drn import drn_d_107, drn_c_26


# Parse Args
parser = argparse.ArgumentParser(description='Inclusive Images Challenge')
parser.add_argument('--lr', default=1, type=float, help='learning rate')
parser.add_argument('--batch-size', default=16, type=float, help='batch size')
parser.add_argument('--exp_name', default='train_without_machine_labels_484', type=str,
                    help='name of experiment')
parser.add_argument('--resume', '-r', action='store_true', default=False,
                    help='resume from checkpoint')
parser.add_argument('--checkpoint-path', default='./checkpoints_all_trainable/checkpoint.pth.tar', type=str,
                    help='name of experiment')
args = parser.parse_args()


# Logging
LOGGER = logging.getLogger(__name__)
log_file = os.path.join('logs', '{}.log'.format(args.exp_name))
os.makedirs('logs', exist_ok=True)
setup_logging(log_path=log_file, logger=LOGGER)


class RandomCrop(object):
    """Takes a random crop of a PIL.Image with given dimensions.

    Attributes:
        size (tuple/int): image dimension after cropping (height, weight)
        , if a single integer i is passed, output dimension is (i,i)

    """

    def __init__(self, size):
        """Constructor."""
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
                 n_trainable_subset=None, reload_labels=False):
        """Constructor."""
        self.data_path = data_path
        self.mode = mode
        self.transform = transform
        self.cache_dir = cache_dir
        self.n_trainable_subset = n_trainable_subset
        if mode != 'test' and mode != 'finetune':
            self.train_images, self.val_images = self.get_train_val_split()
            cache_exists = self.load_labels_from_cache()
            if not cache_exists or reload_labels:
                LOGGER.info("Creating label dicts, this takes a few minutes ...")
                self.trainval_human_labels = self.read_trainval_human_labels()
                self.trainval_machine_labels = self.read_trainval_machine_labels()
                self.trainval_machine_labels = {}
                self.trainval_bbox_labels = self.read_trainval_bbox_labels()
                self.save_labels_to_cache()
        self.test_labels = self.read_tuning_labels_stage_1()
        self.fine_tune_samples = list(self.test_labels.keys())
        self.label_map, self.reverse_label_map = self.get_label_map()
        self.n_trainable_classes = len(self.label_map)
        # Choose if we wish to train on all classes or a subset of the most frequent classes
        if self.n_trainable_subset is None:
            self.n_trainable_subset = (0, self.n_trainable_classes)
            self.classes_subset = list(self.label_map.keys())
        else:
            self.classes_subset = self.get_n_most_frequent_classes(self.test_labels,
                                                                   self.n_trainable_subset[0],
                                                                   self.n_trainable_subset[1])
        if mode != 'test' and mode != 'finetune':
            # Remove samples which don't have a trainable class
            self.filter_set_based_on_trainable_classes(set='train')
            # self.filter_set_based_on_trainable_classes(set='val')
            LOGGER.info("No. of train images: {}, val images: {}".format(len(self.train_images),
                                                                         len(self.val_images)))
            LOGGER.info("No. of trainable classes: {}".format(len(self.classes_subset)))
        self.test_images = self.get_test_image_list()
        if mode == 'test':
            LOGGER.info("No. of test images: {}".format(len(self.test_images)))
            LOGGER.info("No. of trainable classes: {}".format(len(self.classes_subset)))

    def get_n_trainable_classes(self):
        """Return n_trainable_classes attribute."""
        return self.n_trainable_classes

    def get_reverse_label_map(self):
        """Return reverse_label_map attribute."""
        return self.reverse_label_map

    def load_labels_from_cache(self):
        """Load label dicts from cache."""
        human_labels_cache_path = os.path.join(self.cache_dir, 'human_labels.pkl')
        machine_labels_cache_path = os.path.join(self.cache_dir, 'machine_labels.pkl')
        bbox_labels_cache_path = os.path.join(self.cache_dir, 'bbox_labels.pkl')
        if os.path.isfile(human_labels_cache_path) and \
            os.path.isfile(machine_labels_cache_path) and \
                os.path.isfile(machine_labels_cache_path):
            LOGGER.info("Loading label dicts from cache ...")
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
        """Get length of dataset."""
        if self.mode == 'train':
            return len(self.train_images)
        elif self.mode == 'val':
            return len(self.val_images)
        elif self.mode == 'test':
            return len(self.test_images)
        elif self.mode == 'finetune':
            return len(self.fine_tune_samples)

    def get_train_val_split(self):
        """Split training set for competition to train/val."""
        with open("trainval_image_paths.txt", "r") as f:
            trainval_images = f.read().splitlines()
            random.seed(500)
            random.shuffle(trainval_images)
            len_trainval_set = len(trainval_images)
            train_images = trainval_images[0:int(1 * len_trainval_set)]
            val_images = trainval_images[int(0.95 * len_trainval_set):]
        return train_images, val_images

    def filter_set_based_on_trainable_classes(self, set='train'):
        """Remove samples if they don't contain a trainable class."""
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
        LOGGER.info("Size of filtered {} set: {}, {} samples removed".format(set,
                                                                             n_samples_filtered,
                                                                             n_samples_removed))

    def get_class_map_dict(self):
        """Read the class descriptions to a dict."""
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
        label_freqs = list(labels_freq.values())
        label_ids = list(reversed([x for _, x in sorted(zip(label_freqs, label_ids))]))
        trainable_label_ids = label_ids[lower:upper]
        return trainable_label_ids

    def get_class_weights(self):
        """Get class weights."""
        class_weights = np.zeros(self.n_trainable_classes)
        for label_id in self.classes_subset:
            cls_index = self.label_map.get(label_id, None)
            if cls_index is not None:
                class_weights[cls_index] = 1
        class_weights = torch.from_numpy(class_weights).float().to(device)
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
        """Merge list of all annotations."""
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
        """Get one item for data loading."""
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
        one_hot = self.label_to_vector(label)
        return img, one_hot, image_id

    def label_to_vector(self, labels):
        """Convert string of labels to vector."""
        one_hot = np.zeros(self.n_trainable_classes)
        for idx, label in enumerate(labels['labels']):
            # mapped to trainable classes
            if label in self.classes_subset:
                label_index = self.label_map.get(label, None)
                label_confidence = labels['confidences'][idx]
                if label_index is not None:
                    one_hot[label_index] = label_confidence
        return one_hot


class Net(nn.Module):
    """Multi-label Conv-Net classifier."""

    def __init__(self, num_classes=1000):
        """Constructor."""
        super(Net, self).__init__()
        self.drn = drn_d_107(num_classes=num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        """Forward pass."""
        out = self.drn(x)
        out = torch.sigmoid(out)
        return out


class Trainer:
    """Trainer.

    Attributes:
        data_path (str): path to dataset
        n_trainable_subset (int): number of most frequent classes to train on

    """

    def __init__(self, data_path=None, submissions_path='submissions',
                 n_trainable_subset=None, reload_labels=False):
        """Constructor."""
        self.data_path = data_path
        self.n_trainable_subset = n_trainable_subset
        self.submission = self.load_sample_submission()
        self.tuning_labels = self.load_tuning_labels_stage_1()
        self.submissions_path = submissions_path
        self.reload_labels = reload_labels
        os.makedirs(self.submissions_path, exist_ok=True)
        LOGGER.info("Preparing data loaders ...")
        self.prepare_loaders()
        LOGGER.info("Preparing model ...")
        self.model = Net(num_classes=self.num_classes).to(device)
        self.load_checkpoint()
        LOGGER.info("Preparing optimizer ...")
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def load_checkpoint(self):
        """Load checkpoint."""
        if args.resume:
            # Load checkpoint.
            LOGGER.info('Resuming from checkpoint ...')
            assert os.path.isfile(args.checkpoint_path), 'Error: no checkpoint found!'
            checkpoint = torch.load(args.checkpoint_path)
            LOGGER.info("F2-Score: {}".format(checkpoint['f2_score']))
            state_dict = checkpoint['state_dict']
            self.model.load_state_dict(state_dict)

    def load_sample_submission(self):
        """Load sample submission csv file."""
        file_path = os.path.join(self.data_path, 'misc', 'stage_1_sample_submission.csv')
        submission = pd.read_csv(file_path, index_col='image_id')
        return submission

    def load_tuning_labels_stage_1(self):
        """Load the tuning labels (subset of testset)."""
        file_path = os.path.join(self.data_path, 'misc', 'tuning_labels.csv')
        tuning_labels = pd.read_csv(file_path, names=['id', 'labels'], index_col=['id'])
        return tuning_labels

    def get_pre_process(self, phase='train'):
        """Get pre-process function."""
        if phase == 'train':
            return transforms.Compose([transforms.Resize((256, 256)),
                                       transforms.RandomHorizontalFlip(),
                                       RandomCrop((224, 224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.4561, 0.4303, 0.3950),
                                                            (0.1257, 0.1236, 0.1281))])
        else:
            return transforms.Compose([transforms.Resize((256, 256)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.4561, 0.4303, 0.3950),
                                                            (0.1257, 0.1236, 0.1281))])

    def prepare_train_loader(self):
        """Prepare training data-loader."""
        pre_process = self.get_pre_process()
        trainset = IncImagesDataset(self.data_path,
                                    transform=pre_process,
                                    mode='train',
                                    n_trainable_subset=self.n_trainable_subset,
                                    reload_labels=self.reload_labels)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                       collate_fn=custom_collate,
                                                       shuffle=True, num_workers=8)
        self.num_classes = trainset.get_n_trainable_classes()
        self.reverse_label_map = trainset.get_reverse_label_map()
        LOGGER.info("Getting class masking weights ...")
        self.class_weights = trainset.get_class_weights()


    def prepare_finetune_loader(self):
        """Prepare loader for stage-1 finetune set."""
        pre_process = self.get_pre_process()
        finetuneset = IncImagesDataset(self.data_path,
                                       transform=pre_process,
                                       mode='finetune',
                                       reload_labels=self.reload_labels)
        self.finetuneloader = torch.utils.data.DataLoader(finetuneset, batch_size=args.batch_size,
                                                          collate_fn=custom_collate,
                                                          shuffle=False, num_workers=8)

    def prepare_val_loader(self):
        """Prepare validation data-loader."""
        pre_process = self.get_pre_process(phase='val')
        valset = IncImagesDataset(self.data_path,
                                  transform=pre_process,
                                  mode='val',
                                  n_trainable_subset=self.n_trainable_subset,
                                  reload_labels=self.reload_labels)
        self.valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                                     collate_fn=custom_collate,
                                                     shuffle=False, num_workers=8)

    def prepare_test_loader(self):
        """Prepare loader for testing."""
        pre_process = self.get_pre_process(phase='test')
        testset = IncImagesDataset(self.data_path,
                                   transform=pre_process,
                                   mode='test',
                                   reload_labels=self.reload_labels)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                                      collate_fn=custom_collate,
                                                      shuffle=False, num_workers=4)

    def prepare_loaders(self):
        """Prepare data loaders."""
        self.prepare_train_loader()
        self.prepare_finetune_loader()
        # self.prepare_val_loader()
        # self.prepare_test_loader()

    def lower_lr(self):
        """Decrease learning rate by a factor of 10."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10

    def train(self, epoch=0):
        """Train model."""
        LOGGER.info("Train epoch: {}".format(epoch))
        self.model.train()
        train_loss = 0
        score = 0
        n_batches = int(len(self.trainloader.dataset) / args.batch_size)
        for batch_idx, (inputs, targets, _) in enumerate(self.trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = F.binary_cross_entropy(outputs, targets, weight=self.class_weights,
                                          reduction='none').sum() / args.batch_size
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            score += compute_f_score(outputs, targets).item()
            LOGGER.info("Batches done: {}/{} | Loss: {:03f} | F2-Score: {:03f}".format(batch_idx+1,
                                                                                       n_batches,
                                                                                       train_loss/(batch_idx+1),
                                                                                       score/(batch_idx+1)))

    def finetune(self, epoch=0):
        """Finetune model."""
        LOGGER.info("Finetune epoch: {}".format(epoch))
        self.model.train()
        train_loss = 0
        score = 0
        n_batches = int(len(self.finetuneloader.dataset) / args.batch_size)
        for batch_idx, (inputs, targets, _) in enumerate(self.finetuneloader):
            inputs, targets = inputs.to(device), targets.to(device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = F.binary_cross_entropy(outputs, targets, weight=self.class_weights,
                                          reduction='none').sum() / args.batch_size
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            score += compute_f_score(outputs, targets).item()
            LOGGER.info("Batches done: {}/{} | Loss: {:03f} | F2-Score: {:03f}".format(batch_idx+1,
                                                                                       n_batches,
                                                                                       train_loss/(batch_idx+1),
                                                                                       score/(batch_idx+1)))

    def test(self, epoch=0, val=False, save_submission=False, run_on_finetune=False):
        """Test model."""
        self.model.eval()
        if val:
            LOGGER.info("Val epoch: {}".format(epoch))
            loader = self.valloader
        else:
            if run_on_finetune:
                LOGGER.info("Running on test set with labels, save predictions")
                loader = self.finetuneloader
            else:
                LOGGER.info("Running on full test set, save predictions")
                loader = self.testloader
        test_loss = 0
        score = 0
        n_batches = int(len(loader.dataset) / args.batch_size)
        with torch.no_grad():
            for batch_idx, (inputs, targets, image_ids) in enumerate(loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.model(inputs)
                loss = F.binary_cross_entropy(outputs, targets, reduction='none').sum() / args.batch_size
                test_loss += loss.item()
                score += compute_f_score(outputs, targets).item()
                LOGGER.info("Batches done: {}/{} | Loss: {:04f} | F2-Score: {:03f}".format(batch_idx+1,
                                                                                           n_batches,
                                                                                           test_loss/(batch_idx+1),
                                                                                           score/(batch_idx+1)))
                f2_score = score / (batch_idx+1)

                if save_submission:
                    preds = self.convert_outputs_to_label_predictions(outputs)
                    for idx, image_id in enumerate(image_ids):
                        self.submission['labels'][image_id] = preds[idx]

        if save_submission:
            # self.submission.update(self.tuning_labels) # this is useless and fooling yourself
            submission_file_path = os.path.join(self.submissions_path,
                                                'submission_{}_{}.csv'.format(args.exp_name, epoch))
            self.submission.to_csv(submission_file_path)

        return f2_score

    def convert_outputs_to_label_predictions(self, outputs, threshold=0.25):
        """Convert output predictions to label codes."""
        label_preds = []
        preds = (outputs > threshold).cpu().numpy()
        for idx in range(preds.shape[0]):
            label_pred = []
            pred_indices = np.where(preds[idx] == 1)[0]
            for ind in pred_indices:
                label_pred.append(self.reverse_label_map[ind])
            label_preds.append(' '.join(label_pred))
        return label_preds


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        cudnn.benchmark = True
    os.environ['TORCH_HOME'] = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                            'torchvision')
    trainer = Trainer(data_path='/staging/inc_images', n_trainable_subset=(0, 484),
                      reload_labels=False)
    best_score, is_best, score = 0, False, 0
    LOGGER.info("Starting training ...")
    for epoch in range(5):
        trainer.train(epoch=epoch)
        score = trainer.test(epoch=epoch, run_on_finetune=True)
        LOGGER.info("F2-Score: {}".format(score))
        if score > best_score:
            is_best = True
            best_score = score
        else:
            is_best = False
        save_checkpoint({'epoch': epoch + 1, 'f2_score': score,
                         'state_dict': trainer.model.state_dict()},
                        args.exp_name,
                        backup_as_best=is_best)

    # NOTE: Finetune
    # trainer.lower_lr()
    # for i in range(50):
    #     trainer.finetune(epoch=epoch)
    # score = trainer.test(epoch=110, save_submission=True)
