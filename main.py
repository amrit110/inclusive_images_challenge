"""Main script for inclusive images challenge."""

import argparse, os, glob, random, collections, copy
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


# Parse Args
parser = argparse.ArgumentParser(description='Inclusive Images Challenge')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch-size', default=32, type=float, help='batch size')
parser.add_argument('--exp_name', default='inclusive_images_challenge', type=str,
                    help='name of experiment')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()


# Logging
LOGGER = logging.getLogger(__name__)
log_file = os.path.join('logs', '{}.log'.format(args.exp_name))
os.makedirs('logs', exist_ok=True)
setup_logging(log_path=log_file, logger=LOGGER)


class IncImagesDataset:
    """Dataset for the challenge."""

    def __init__(self, data_path, mode='train', transform=None, cache_dir='cache'):
        """Constructor."""
        self.data_path = data_path
        self.mode = mode
        self.transform = transform
        self.cache_dir = cache_dir
        cache_exists = self.load_labels_from_cache()
        if not cache_exists:
            LOGGER.info("Creating label dicts, this takes a few minutes ...")
            self.trainval_human_labels = self.read_trainval_human_labels()
            self.trainval_machine_labels = self.read_trainval_machine_labels()
            self.trainval_bbox_labels = self.read_trainval_bbox_labels()
            self.save_labels_to_cache()
        self.test_labels = self.read_tuning_labels_stage_1()
        self.train_images, self.val_images = self.get_train_val_split()
        self.label_map, self.reverse_label_map = self.get_label_map()
        self.n_trainable_classes = len(self.label_map)
        if mode == 'train':
            LOGGER.info("No. of train images: {}, val images: {}".format(len(self.train_images),
                                                                         len(self.val_images)))
            LOGGER.info("No. of trainable classes: {}".format(self.n_trainable_classes))
        self.test_images = self.get_test_image_list()
        if mode == 'test':
            LOGGER.info("No. of test images: {}".format(len(self.test_images)))
            LOGGER.info("No. of trainable classes: {}".format(self.n_trainable_classes))

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

    def get_train_val_split(self):
        """Split training set for competition to train/val."""
        with open("trainval_image_paths.txt", "r") as f:
            trainval_images = f.read().splitlines()
            random.seed(500)
            random.shuffle(trainval_images)
            len_trainval_set = len(trainval_images)
            train_images = trainval_images[0:int(0.8 * len_trainval_set)]
            val_images = trainval_images[int(0.8 * len_trainval_set):]
        return train_images, val_images

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
        contents = read_csv(file_path)
        for idx, item in enumerate(contents):
            image_id = item[0]
            labels = item[1].split(' ')
            tuning_labels[image_id] = labels
        return tuning_labels

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
        img = Image.open(image_path)
        img = self.transform(*[img])
        image_id = image_path.split('/')[-1].split('.')[0]
        if self.mode == 'train' or 'val':
            label_human = self.trainval_human_labels.get(image_id,
                                                         {'labels': [None], 'confidences': [None]})
            label_machine = self.trainval_machine_labels.get(image_id,
                                                             {'labels': [None], 'confidences': [None]})
            label_bbox = self.trainval_bbox_labels.get(image_id,
                                                       {'labels': [None], 'confidences': [None]})
            label = self.merge_labels(label_human, label_machine, label_bbox)
        elif self.mode == 'test':
            label = self.test_labels[image_id]
        one_hot = self.label_to_one_hot(label)
        return img, one_hot

    def label_to_one_hot(self, labels):
        """Convert string of labels to one-hot vector."""
        one_hot = np.zeros(self.n_trainable_classes)
        for idx, label in enumerate(labels['labels']):
            # mapped to trainable classes
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
        self.resnet = models.resnet152(pretrained=False)
        self.num_classes = num_classes
        self.redefine_final_layer()

    def redefine_final_layer(self):
        """Make final layer have correct number of feature maps."""
        last_layer_replace = nn.Linear(2048, self.num_classes)
        self.resnet._modules['fc'] = last_layer_replace

    def forward(self, x):
        """Forward pass."""
        out = self.resnet(x)
        out = torch.sigmoid(out)
        return out


class Trainer:
    """Trainer."""

    def __init__(self):
        """Constructor."""
        LOGGER.info("Preparing data loaders ...")
        self.prepare_loaders()
        LOGGER.info("Preparing model ...")
        self.model = Net(num_classes=self.num_classes).to(device)
        LOGGER.info("Preparing optimizer ...")
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01,
                                         momentum=0.9)

    def prepare_loaders(self):
        """Prepare data loaders."""
        pre_process_train = transforms.Compose([transforms.Resize((224, 224)),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4561, 0.4303, 0.3950),
                                                                     (0.1257, 0.1236, 0.1281))])
        pre_process_val = transforms.Compose([transforms.Resize((224, 224)),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.4561, 0.4303, 0.3950),
                                                                   (0.1257, 0.1236, 0.1281))])
        pre_process_test = transforms.Compose([transforms.Resize((224, 224)),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.4561, 0.4303, 0.3950),
                                                                    (0.1257, 0.1236, 0.1281))])
        trainset = IncImagesDataset('/staging/inc_images',
                                    transform=pre_process_train, mode='train')
        self.num_classes = trainset.n_trainable_classes
        valset = IncImagesDataset('/staging/inc_images',
                                  transform=pre_process_val, mode='val')
        testset = IncImagesDataset('/staging/inc_images',
                                   transform=pre_process_test, mode='test')
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                       collate_fn=custom_collate,
                                                       shuffle=True, num_workers=8)
        self.valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                                     collate_fn=custom_collate,
                                                     shuffle=False, num_workers=4)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                                      collate_fn=custom_collate,
                                                      shuffle=False, num_workers=4)

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
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = F.binary_cross_entropy(outputs, targets)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            score += compute_f_score(outputs, targets).item()
            LOGGER.info("Batches done: {}/{} | Loss: {:03f} | F2-Score: {:03f}".format(batch_idx+1,
                                                                                       n_batches,
                                                                                       train_loss/(batch_idx+1),
                                                                                       score/(batch_idx+1)))

    def test(self, epoch=0, val=False):
        """Test model."""
        self.model.eval()
        if val:
            LOGGER.info("Val epoch: {}".format(epoch))
            loader = self.valloader
        else:
            LOGGER.info("Test epoch: {}".format(epoch))
            loader = self.testloader
        test_loss = 0
        score = 0
        n_batches = int(len(loader.dataset) / args.batch_size)
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.model(inputs)

                loss = F.binary_cross_entropy(outputs, targets)
                test_loss += loss.item()
                score += compute_f_score(outputs, targets).item()
                LOGGER.info("Batches done: {}/{} | Loss: {:04f} | F2-Score: {:03f}".format(batch_idx+1,
                                                                                           n_batches,
                                                                                           test_loss/(batch_idx+1),
                                                                                           score/(batch_idx+1)))
        return score


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        cudnn.benchmark = True
    os.environ['TORCH_HOME'] = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                            'torchvision')
    trainer = Trainer()
    best_score, is_best, score = 0, 0, False
    LOGGER.info("Starting training ...")
    for epoch in range(50):
        trainer.train()
        trainer.test(val=True)
        score = trainer.test()
        if score > best_score:
            is_best = True
            best_score = score
        save_checkpoint({'epoch': epoch + 1, 'f2_score': score,
                         'state_dict': trainer.model.state_dict()},
                        'checkpoints',
                        backup_as_best=is_best)
        if (epoch + 1) % 10:
            trainer.lower_lr()