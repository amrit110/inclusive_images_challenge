"""Module to implement a trainer class."""

import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torchvision.transforms as transforms

from lib.dataset import IncImagesDataset, RandomCrop
from lib.utils import custom_collate, wrap_cuda, compute_f_score
from lib.model import CNN


class Trainer:
    """Trainer.

    Attributes:
        args (AttrDict): arguments passed from main script.
        logger (logging.Logger): logger.
        num_classes (int): number of trainable classes.
        class_weights (torch.Tensor): class weights used in loss function.

    """

    def __init__(self, args, logger=None):
        """Constructor.

        Args:
            args (AttrDict): arguments passed from main script.
            logger (logging.Logger): logger.

        """
        self.args = args
        self.logger = logger

        os.makedirs(self.args.submissions_path, exist_ok=True)

        self.logger.info("Preparing data loaders ...")
        self.prepare_loaders()

        self.logger.info("Preparing model ...")
        self.model = wrap_cuda(CNN(num_classes=self.num_classes))
        self.load_checkpoint()

        self.logger.info("Preparing optimizer ...")
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def load_checkpoint(self):
        """Load checkpoint."""
        if self.args.resume:
            # Load checkpoint.
            self.logger.info('Resuming from checkpoint ...')
            assert os.path.isfile(self.args.checkpoint_path), 'Error: no checkpoint found!'
            checkpoint = torch.load(self.args.checkpoint_path)

            self.logger.info("F2-Score: {}".format(checkpoint['f2_score']))
            state_dict = checkpoint['state_dict']
            self.model.load_state_dict(state_dict)

    def load_sample_submission(self):
        """Load sample submission csv file."""
        file_path = os.path.join(self.args.data_path, 'misc', 'stage_1_sample_submission.csv')
        submission = pd.read_csv(file_path, index_col='image_id')

        return submission

    def load_tuning_labels_stage_1(self):
        """Load the tuning labels (subset of testset)."""
        file_path = os.path.join(self.args.data_path, 'misc', 'tuning_labels.csv')
        tuning_labels = pd.read_csv(file_path, names=['id', 'labels'], index_col=['id'])

        return tuning_labels

    def get_pre_process(self, phase='train'):
        """Get pre-process function."""
        crop_size = int(self.args.img_size - 2 ** (math.log2(self.args.img_size) - 3)) # hack
        if phase == 'train':
            return transforms.Compose([transforms.Resize((self.args.img_size, self.args.img_size)),
                                       transforms.RandomHorizontalFlip(),
                                       RandomCrop((crop_size, crop_size)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.4561, 0.4303, 0.3950),
                                                            (0.1257, 0.1236, 0.1281))])
        else:
            return transforms.Compose([transforms.Resize((self.args.img_size, self.args.img_size)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.4561, 0.4303, 0.3950),
                                                            (0.1257, 0.1236, 0.1281))])

    def prepare_train_loader(self):
        """Prepare training data-loader."""
        pre_process = self.get_pre_process()
        trainset = IncImagesDataset(self.args.data_path,
                                    transform=pre_process,
                                    mode='train',
                                    n_trainable_subset=self.args.n_trainable_subset,
                                    reload_labels=self.args.reload_labels,
                                    logger=self.logger)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.args.batch_size,
                                                       collate_fn=custom_collate,
                                                       shuffle=True, num_workers=8)
        self.num_classes = trainset.get_n_trainable_classes()
        self.reverse_label_map = trainset.get_reverse_label_map()

        self.logger.info("Getting class masking weights ...")
        self.class_weights = trainset.get_class_weights()


    def prepare_finetune_loader(self):
        """Prepare loader for stage-1 finetune set."""
        pre_process = self.get_pre_process()
        finetuneset = IncImagesDataset(self.args.data_path,
                                       transform=pre_process,
                                       mode='finetune',
                                       reload_labels=self.args.reload_labels,
                                       logger=self.logger)
        self.finetuneloader = torch.utils.data.DataLoader(finetuneset,
                                                          batch_size=self.args.batch_size,
                                                          collate_fn=custom_collate,
                                                          shuffle=False, num_workers=8)

    def prepare_val_loader(self):
        """Prepare validation data-loader."""
        pre_process = self.get_pre_process(phase='val')
        valset = IncImagesDataset(self.args.data_path,
                                  transform=pre_process,
                                  mode='val',
                                  n_trainable_subset=self.args.n_trainable_subset,
                                  reload_labels=self.args.reload_labels,
                                  logger=self.logger)
        self.valloader = torch.utils.data.DataLoader(valset, batch_size=self.args.batch_size,
                                                     collate_fn=custom_collate,
                                                     shuffle=False, num_workers=8)

    def prepare_test_loader(self):
        """Prepare loader for testing."""
        pre_process = self.get_pre_process(phase='test')
        testset = IncImagesDataset(self.args.data_path,
                                   transform=pre_process,
                                   mode='test',
                                   reload_labels=self.args.reload_labels,
                                   logger=self.logger)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=self.args.batch_size,
                                                      collate_fn=custom_collate,
                                                      shuffle=False, num_workers=4)

    def prepare_loaders(self):
        """Prepare data loaders."""
        self.prepare_train_loader()
        self.prepare_finetune_loader()
        # self.prepare_val_loader()
        self.prepare_test_loader()

    def lower_lr(self):
        """Decrease learning rate by a factor of 10."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10

    def train(self, epoch=0):
        """Train model."""
        self.logger.info("Train epoch: {}".format(epoch))
        self.model.train()
        train_loss = 0
        score = 0
        n_batches = int(len(self.trainloader.dataset) / self.args.batch_size)

        for batch_idx, (inputs, targets, _) in enumerate(self.trainloader):
            inputs, targets = wrap_cuda(inputs), wrap_cuda(targets)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = F.binary_cross_entropy(outputs, targets, weight=self.class_weights,
                                          reduction='none').sum() / self.args.batch_size
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            score += compute_f_score(outputs, targets).item()
            self.logger.info("Batches done: {}/{} | Loss: {:03f} | F2-Score: {:03f}".format(batch_idx+1,
                                                                                       n_batches,
                                                                                       train_loss/(batch_idx+1),
                                                                                       score/(batch_idx+1)))

    def finetune(self, epoch=0):
        """Finetune model."""
        self.logger.info("Finetune epoch: {}".format(epoch))
        self.model.train()
        train_loss = 0
        score = 0
        n_batches = int(len(self.finetuneloader.dataset) / self.args.batch_size)

        for batch_idx, (inputs, targets, _) in enumerate(self.finetuneloader):
            inputs, targets = wrap_cuda(inputs), wrap_cuda(targets)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = F.binary_cross_entropy(outputs, targets, weight=self.class_weights,
                                          reduction='none').sum() / self.args.batch_size
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            score += compute_f_score(outputs, targets).item()
            self.logger.info("Batches done: {}/{} | Loss: {:03f} | F2-Score: {:03f}".format(batch_idx+1,
                                                                                       n_batches,
                                                                                       train_loss/(batch_idx+1),
                                                                                       score/(batch_idx+1)))

    def test(self, epoch=0, val=False, save_submission=False, run_on_finetune=False):
        """Test model."""
        self.model.eval()
        if val:
            self.logger.info("Val epoch: {}".format(epoch))
            loader = self.valloader
        else:
            if run_on_finetune:
                self.logger.info("Running on test set with labels, save predictions")
                loader = self.finetuneloader
            else:
                self.logger.info("Running on full test set, save predictions")
                loader = self.testloader
        test_loss = 0
        score = 0
        n_batches = int(len(loader.dataset) / self.args.batch_size)

        with torch.no_grad():
            for batch_idx, (inputs, targets, image_ids) in enumerate(loader):
                inputs, targets = wrap_cuda(inputs), wrap_cuda(targets)
                outputs = self.model(inputs)
                loss = F.binary_cross_entropy(outputs, targets, reduction='none').sum() / self.args.batch_size
                test_loss += loss.item()
                score += compute_f_score(outputs, targets).item()
                self.logger.info("Batches done: {}/{} | Loss: {:04f} | F2-Score: {:03f}".format(batch_idx+1,
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
            submission_file_path = os.path.join(self.args.submissions_path,
                                                'submission_{}_{}.csv'.format(self.args.exp_name, epoch))
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
