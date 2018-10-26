"""Main script for inclusive images challenge."""


# Imports
import argparse, os, collections, copy
from attrdict import AttrDict
import math
import logging

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from lib.utils import setup_logging, save_checkpoint
from lib.trainer import *


# Parse args.
parser = argparse.ArgumentParser(description='Inclusive Images Challenge')
parser.add_argument('--lr', default=1, type=float, help='learning rate')
parser.add_argument('--batch-size', default=64, type=float, help='batch size')
parser.add_argument('--img-size', default=128, type=float, help='image size to use')
parser.add_argument('--n-trainable-subset', default=None, type=tuple,
                    help='number of top subset classes to train')
parser.add_argument('--exp-name', default='debug', type=str,
                    help='name of experiment')
parser.add_argument('--resume', '-r', action='store_true', default=False,
                    help='resume from checkpoint')
parser.add_argument('--checkpoint-path',
                    default='./experiments/ensemble_stage_1_5/model_best.pth.tar',
                    type=str,
                    help='path to save checkpoint')
parser.add_argument('--data-path', default='/staging/inc_images', type=str,
                    help='path to stored data')
parser.add_argument('--submissions-path', default='./submissions', type=str,
                    help='path to save submission files')
parser.add_argument('--reload-labels', default=False, type=bool,
                    help='flag if labels need to re-created, if false, will load from cache')
parser.add_argument('--mode', default='train', type=str,
                    choices=['train', 'finetune', 'test', 'adapt'], help='train or test mode')
parser.add_argument('--use-ensemble', default=False, type=bool,
                    help='use ensemble of networks')
parser.add_argument('--n-models', default=1, type=int,
                    help='number of models in ensemble')
parser.add_argument('--ensemble-checkpoint-paths',
                    default=['experiments/ensemble_stage_1_1_f/checkpoint.pth.tar',
                             'experiments/ensemble_stage_1_2_f/checkpoint.pth.tar',
                             'experiments/ensemble_stage_1_3_f/checkpoint.pth.tar',
                             'experiments/ensemble_stage_1_5_f/checkpoint.pth.tar'],
                    type=list,
                    help='number of models in ensemble')

args = parser.parse_args()

# Convert to int.
args.batch_size = int(args.batch_size)
args.img_size = int(args.img_size)


# Additional augmentation settings.
aug_settings = {
    "distort_pixels": {
        "use_prob": 0.5,
        "ops": {
            "gauss_blur": {
                "use_prob": 0.0,
                "sigma": [0, 0.5]
            },
            "contrast": {
                "use": True,
                "alpha": [0.75, 1.5]
            },
            "add_gauss_noise": {
                "use": True,
                "loc": 0,
                "scale": [0.0, 12.75],
                "per_channel": 0.5
            },
            "brightness": {
                "use": True,
                "scale": [0.8, 1.2],
                "per_channel": 0.2
            },
            "hue_saturation": {
                "use": True,
                "min": -10,
                "max": 10
            }
        }
    }
}
args.aug_settings = AttrDict(aug_settings) # Add to args.


# Logging
LOGGER = logging.getLogger(__name__)
log_file = os.path.join('logs', '{}.log'.format(args.exp_name))
os.makedirs('logs', exist_ok=True)
setup_logging(log_path=log_file, logger=LOGGER)


if __name__ == '__main__':
    # Experiment directory.
    experiment_path = os.path.join('experiments', args.exp_name)
    os.makedirs(experiment_path, exist_ok=True)


    # Set cudnn profiling, add torchvision path.
    if torch.cuda.is_available():
        cudnn.benchmark = True
    os.environ['TORCH_HOME'] = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                            'torchvision')

    # Initialise trainer.
    trainer = CNNTrainer(args, logger=LOGGER)


    best_score, is_best, score = 0, False, 0
    # NOTE: 'train', 'finetune' and 'test' modes support single models, and
    # adapt ensemble expects to load single model weights as checkpoints and
    # create an ensemble, adapt it to a target test set using bootstrapping and
    # generate predictions.

    if args.mode == 'train':
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
                            experiment_path,
                            backup_as_best=is_best)

    elif args.mode == 'finetune':
        LOGGER.info("Starting fine-tuning ...")
        trainer.lower_lr()
        for epoch in range(100):
            trainer.finetune(epoch=epoch)
            save_checkpoint({'epoch': epoch + 1, 'f2_score': 0,
                             'state_dict': trainer.model.state_dict()},
                            experiment_path,
                            backup_as_best=False)

    elif args.mode == 'test':
        LOGGER.info("Starting inference on test set ...")
        score = trainer.test(epoch='final', save_submission=True)

    elif args.mode == 'adapt':
        LOGGER.info("Starting adaptation using bootstapping on test set ...")
        trainer.lower_lr()
        for epoch in range(10):
            trainer.adapt_ensemble(epoch)
