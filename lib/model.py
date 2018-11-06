"""Implement model."""


# Imports.
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.dilated_resnet import drn_d_107
from lib.utils import wrap_cuda


class ModelEnsemble(nn.Module):
    """Ensemble of models.

    Attributes:
        n_models (int): number of models to use in ensemble.
        ensemble (list): list of models.

    """

    def __init__(self, net, num_classes, n_models):
        """Constructor.

        Args:
            net (class): network model class name.
            num_classes (int): number of output channels.
            n_models (int): number of models to use in ensemble.

        """
        super(ModelEnsemble, self).__init__()
        self.n_models = n_models
        self.ensemble = nn.ModuleList([net(num_classes) \
            for i in range(n_models)])

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): input.

        """
        return [model(x) for model in self.ensemble]

    def get_ensemble_average(self, outputs):
         """Concatenate list of output tensors from ensemble, apply softmax.

         Args:
             outputs (list): outputs from model ensemble.

         Returns:
             average_prediction (torch.Tensor): predicted average class probabilities.

         """
         predictions = [output for output in outputs]
         predictions = torch.stack(predictions, dim=1)
         average_prediction = predictions.mean(1)

         return average_prediction

    def parameters(self):
        """Parameters to optimize.

        Yields:
            (list of torch.nn.Parameter): parameters to optimize

        """
        for model in self.ensemble:
            for param in model.parameters():
                yield param

    def train(self):
        """Set each model's training attribute to True (affects batchnorm, dropout)."""
        for model in self.ensemble:
            model.train()

    def eval(self):
        """Set each model's training attribute to False (affects batchnorm, dropout)."""
        for model in self.ensemble:
            model.eval()

    def cuda(self):
        """Transfer to GPU."""
        for idx, _ in enumerate(self.ensemble):
            self.ensemble[idx] = wrap_cuda(self.ensemble[idx])

    def load_checkpoint_list(self, checkpoint_paths):
        """Given list of checkpoints for each model in ensemble, load
        weights."""
        for idx, path in enumerate(checkpoint_paths):
            assert os.path.isfile(path), 'Error: path is wrong!'
            checkpoint = torch.load(path)
            state_dict = checkpoint['state_dict']
            self.ensemble[idx].load_state_dict(state_dict)


class CNN(nn.Module):
    """Multi-label Conv-Net classifier."""

    def __init__(self, num_classes):
        """Constructor."""
        super(CNN, self).__init__()
        self.drn = drn_d_107(num_classes=num_classes, pool_size=14)
        self.num_classes = num_classes

    def forward(self, x):
        """Forward pass."""
        out = self.drn(x)
        output = torch.sigmoid(out[0])

        return output
