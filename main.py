"""Main script for inclusive image challenge."""

import os, csv
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models


def read_csv(file_path):
    """Read csv file."""
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        out_list = list(reader)
        return out_list


class IncImagesDataset:
    """Dataset for the challenge."""

    def __init__(self, data_path):
        """Constructor."""
        self.data_path = data_path

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
        class_map_dict = get_class_map_dict()
        trainable_classes = {}
        for item in contents:
            label_code = item[0]
            if label_code in class_map_dict:
                trainable_classes[label_code] = class_map_dict[label_code]
        return trainable_classes

    def read_train_human_labels(self):
        """Read the training labels annotated by humans."""
        human_train_labels = {}
        file_path = os.path.join(self.data_path, 'misc', 'train_human_labels.csv')
        contents = read_csv(file_path)
        class_map = get_class_map_dict()
        for item in contents:
            image_id = item[0]
            source = item[1]
            label_code = item[2]
            confidence = item[3]
            if label_code in class_map:
                if not human_train_labels.get(image_id):
                    human_train_labels[image_id] = {'labels': [], 'confidences': []}
                else:
                    human_train_labels[image_id]['labels'].append(class_map[label_code])
                    human_train_labels[image_id]['confidences'].append(confidence)
        return human_train_labels

    def get_test_image_list(self):
        """Get list of paths to test images."""
        images_path = os.path.join(self.data_path, 'misc', 'stage_1_test_images')
        images = glob.glob(os.path.join(images_path, '*.jpg'))
        return images

    def write_train_image_list(self):
        """Get list of paths to train images, write them to txt file."""
        train_image_list = []
        folder_list = ['train_0', 'train_1', 'train_2', 'train_3', 'train_4',
                       'train_5', 'train_6', 'train_7', 'train_8', 'train_9',
                       'train_a', 'train_b', 'train_c', 'train_d', 'train_e',
                       'train_f']
        for folder in folder_list:
            images_path = os.path.join(data_path, 'train_images', folder)
            train_image_list.extend(glob.glob(os.path.join(images_path, '*.jpg')))
        with open('train_image_paths.txt', 'w') as f:
            for item in my_list:
                f.write("%s\n" % item)


class Net(nn.Module):
    """Multi-label Conv-Net classifier."""

    def __init__(self, num_classes=1000):
        """Constructor."""
        super(Net, self).__init__()
        self.resnet = models.resnet152()
        self.num_classes = num_classes
        self.redefine_final_layer()

    def redefine_final_layer(self):
        """Make final layer have correct number of feature maps."""
        last_layer_replace = nn.Linear(2048, self.num_classes)
        self.resnet._modules['fc'] = last_layer_replace

    def forward(self, x):
        """Forward pass."""
        pass


class Trainer:
    """Trainer."""

    def __init__(self):
        """Constructor."""
        self.model = Net(num_classes=7178)



if __name__ == '__main__':
    trainer = Trainer()
    trainer.write_train_image_list()
