"""Module to implement trainer."""


# Imports.
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torchvision.transforms as transforms

from lib.dataset import IncImagesDataset, RandomCrop, DistortPixels
from lib.utils import custom_collate, wrap_cuda, compute_f_score
from lib.model import *


STATUS_MSG = "Batches done: {}/{} | Loss: {:04f} | F2-Score: {:03f}"


class Trainer:
    """Trainer.

    Attributes:
        args (AttrDict): arguments passed from main script.
        logger (logging.Logger): logger.
        num_classes (int): number of trainable classes.
        class_weights (torch.Tensor): class weights used in loss function.
        trainloader (torch.utils.data.DataLoader): training data loader.
        finetuneloader (torch.utils.data.DataLoader): stage-1 data with labels data loader.
        testloader (torch.utils.data.DataLoader): test data loader.
        reverse_label_map (dict): dict to map label indices to label IDs.
        model (CNN): model.
        optimizer (torch.optim.Optimizer): optimizer.
        tuning_labels (dict): stage-1 finetuning labels.

    """

    def __init__(self, args, logger=None):
        """Constructor.

        Args:
            args (AttrDict): arguments passed from main script.
            logger (logging.Logger): logger.

        """
        self.args = args
        self.logger = logger
        self.num_classes = None
        self.trainloader = None
        self.finetuneloader = None
        self.testloader = None
        self.reverse_label_map = None
        self.model = None
        self.optimizer = None
        self.submission = None
        self.tuning_labels = self.load_tuning_labels_stage_1()

        # Load sample submission file to use as format for submission.
        os.makedirs(self.args.submissions_path, exist_ok=True)
        self.submission = self.load_sample_submission()

        self.logger.info("Preparing data loaders ...")
        self.prepare_loaders()

        self.logger.info("Preparing model ...")
        self.prepare_model()

        self.logger.info("Preparing optimizer ...")
        self.prepare_optimizer()

    def prepare_model(self):
        """Prepare model."""
        return NotImplemented

    def prepare_optimizer(self):
        """Prepare optimizer."""
        return NotImplemented

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
        """Get pre-process function.

        Args:
            phase (str, optional): phase, 'train' uses different
            pre-processing, i.e augmentations, while testing does not.

        Returns:
            (torchvision.transforms.transforms.Compose): transforms to be
            applied to image.

        """
        # Crop size smaller than image size, by multiple of 2.
        crop_size = int(self.args.img_size - 2 ** (math.log2(self.args.img_size) - 3))

        if phase == 'train':
            return transforms.Compose([transforms.Resize((self.args.img_size, self.args.img_size)),
                                       transforms.RandomHorizontalFlip(),
                                       RandomCrop((crop_size, crop_size)),
                                       DistortPixels(self.args.aug_settings.distort_pixels.use_prob,
                                                     self.args.aug_settings.distort_pixels.ops),
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

    def prepare_finetune_loader(self, bootstrap_mode=False):
        """Prepare loader for stage-1 finetune set.

        Args:
            bootstrap_mode (str, optional): used for bootstrapping while
            adapting a model to test set.

        """
        pre_process = self.get_pre_process()
        finetuneset = IncImagesDataset(self.args.data_path,
                                       transform=pre_process,
                                       mode='finetune',
                                       reload_labels=self.args.reload_labels,
                                       bootstrap_mode=bootstrap_mode,
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

    def convert_outputs_to_label_predictions(self, outputs, threshold=0.25):
        """Convert output predictions to label codes.

        Args:
            outputs (torch.Tensor): output predictions.
            threshold (float, optional): confidence threshold for predictions.

        Returns:
            (str): label predictions to write to submission file.

        """
        label_preds = []
        preds = (outputs > threshold).cpu().numpy()

        for idx in range(preds.shape[0]):
            label_pred = []
            pred_indices = np.where(preds[idx] == 1)[0]
            for ind in pred_indices:
                label_pred.append(self.reverse_label_map[ind])
            label_preds.append(' '.join(label_pred))

        return label_preds

    def train(self, epoch=0):
        """Train model.

        Args:
            epoch (int, optional): current epoch number.

        """
        return NotImplemented

    def finetune(self, epoch=0):
        """Finetune model.

        Args:
            epoch (int, optional): current epoch number.

        """
        return NotImplemented

    def test(self, epoch=0, val=False, save_submission=False, run_on_finetune=False):
        """Test model.

        Args:
            epoch (int, optional): current epoch number.
            val (bool, otptional): validation mode.
            save_submission (bool, optional): toggle saving submission file.
            run_on_finetune (bool, optional): toggle using finetune data
            loader, i.e stage-1 images with labels..

        """
        return NotImplemented


class CNNTrainer(Trainer):
    """Trainer for baseline CNN."""

    def __init__(self, args, logger=None):
        """Constructor."""
        super(CNNTrainer, self).__init__(args, logger)

    def prepare_model(self):
        """Prepare model."""
        if self.args.use_ensemble:
            self.model = ModelEnsemble(CNN, self.num_classes,
                                       self.args.n_models)
            self.model.cuda()
            self.model.load_checkpoint_list(self.args.ensemble_checkpoint_paths)
        else:
            self.model = wrap_cuda(CNN(self.num_classes))
            self.load_checkpoint()

    def prepare_optimizer(self):
        """Prepare optimizer."""
        self.optimizer = torch.optim.Adam(self.model.parameters())

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
            batch_size = outputs.shape[0]
            loss = F.binary_cross_entropy(outputs, targets,
                                          reduction='none').sum() / batch_size
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            score += compute_f_score(outputs, targets).item()
            self.logger.info(STATUS_MSG.format(batch_idx+1,
                                               n_batches,
                                               train_loss/(batch_idx+1),
                                               score/(batch_idx+1)))

    def finetune(self, epoch=0, model=None):
        """Finetune model."""
        self.logger.info("Finetune epoch: {}".format(epoch))
        if model is None:
            model = self.model
        model.train()

        train_loss = 0
        score = 0
        n_batches = int(len(self.finetuneloader.dataset) / self.args.batch_size)

        for batch_idx, (inputs, targets, _) in enumerate(self.finetuneloader):
            inputs, targets = wrap_cuda(inputs), wrap_cuda(targets)
            self.optimizer.zero_grad()
            outputs = model(inputs)
            batch_size = outputs.shape[0]
            loss = F.binary_cross_entropy(outputs, targets,
                                          reduction='none').sum() / batch_size
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            score += compute_f_score(outputs, targets).item()
            self.logger.info(STATUS_MSG.format(batch_idx+1,
                                               n_batches,
                                               train_loss/(batch_idx+1),
                                               score/(batch_idx+1)))

    def adapt_ensemble(self, epoch=0):
        """Run ensemble inference over test set, bootstrap."""
        self.logger.info("Adaptating ensemble epoch {}".format(epoch))
        self.generate_predictions()
        self.prepare_finetune_loader(bootstrap_mode=True)

        for model in self.model.ensemble:
            self.finetune(epoch=epoch, model=model)

    def generate_predictions(self):
        self.model.eval()
        n_batches = int(len(self.testloader.dataset) / self.args.batch_size)
        with torch.no_grad():
            for batch_idx, (inputs, _, image_ids) in enumerate(self.testloader):
                self.logger.info("Predicting batch {}/{}".format(batch_idx,
                                                                 n_batches))
                inputs = wrap_cuda(inputs)
                outputs = self.model(inputs)
                avg_preds = self.model.get_ensemble_average(outputs)
                preds = self.convert_outputs_to_label_predictions(avg_preds)
                for idx, image_id in enumerate(image_ids):
                    self.submission['labels'][image_id] = preds[idx]

            # NOTE: Use only for stage-1 test set
            self.submission.update(self.tuning_labels)
            submission_file_path = 'pseudo_labels.csv'
            self.submission.to_csv(submission_file_path)

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
                batch_size = outputs.shape[0]
                loss = F.binary_cross_entropy(outputs, targets,
                                              reduction='none').sum() / batch_size
                test_loss += loss.item()
                score += compute_f_score(outputs, targets).item()
                self.logger.info(STATUS_MSG.format(batch_idx+1,
                                                   n_batches,
                                                   test_loss/(batch_idx+1),
                                                   score/(batch_idx+1)))
                f2_score = score / (batch_idx+1)

                if save_submission:
                    preds = self.convert_outputs_to_label_predictions(outputs)
                    for idx, image_id in enumerate(image_ids):
                        self.submission['labels'][image_id] = preds[idx]

        if save_submission:
            self.submission.update(self.tuning_labels)
            submission_file_path = os.path.join(self.args.submissions_path,
                                                'submission_{}_{}.csv'.format(self.args.exp_name, epoch))
            self.submission.to_csv(submission_file_path)

        return f2_score
