"""Main script for inclusive images challenge."""

import argparse, os, collections, copy
import math
import logging

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from lib.utils import setup_logging, save_checkpoint
from lib.trainer import *


# Parse Args
parser = argparse.ArgumentParser(description='Inclusive Images Challenge')
parser.add_argument('--lr', default=1, type=float, help='learning rate')
parser.add_argument('--batch-size', default=64, type=float, help='batch size')
parser.add_argument('--img-size', default=128, type=float, help='image size to use')
parser.add_argument('--n-trainable-subset', default=None, type=tuple,
                    help='number of top subset classes to train')
parser.add_argument('--exp_name', default='cnn_rnn', type=str,
                    help='name of experiment')
parser.add_argument('--resume', '-r', action='store_true', default=False,
                    help='resume from checkpoint')
parser.add_argument('--checkpoint-path',
                    default='./experiments/train_with_machine_labels_484/model_best.pth.tar',
                    type=str,
                    help='path to save checkpoint')
parser.add_argument('--data-path', default='/staging/inc_images', type=str,
                    help='path to stored data')
parser.add_argument('--submissions-path', default='./submissions', type=str,
                    help='path to save submission files')
parser.add_argument('--reload-labels', default=False, type=bool,
                    help='flag if labels need to re-created, if false, will load from cache')
parser.add_argument('--mode', default='train', type=str,
                    choices=['train', 'finetune', 'test'], help='train or test mode')
args = parser.parse_args()

# Convert to int
args.batch_size = int(args.batch_size)
args.img_size = int(args.img_size)


# Logging
LOGGER = logging.getLogger(__name__)
log_file = os.path.join('logs', '{}.log'.format(args.exp_name))
os.makedirs('logs', exist_ok=True)
setup_logging(log_path=log_file, logger=LOGGER)


if __name__ == '__main__':
    experiment_path = os.path.join('experiments', args.exp_name)
    os.makedirs(experiment_path, exist_ok=True)

    if torch.cuda.is_available():
        cudnn.benchmark = True
    os.environ['TORCH_HOME'] = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                            'torchvision')

    # trainer = CNNTrainer(args, logger=LOGGER)
    trainer = CNNRNNTrainer(args, logger=LOGGER)
    best_score, is_best, score = 0, False, 0

    if args.mode == 'train':
        LOGGER.info("Starting training ...")
        for epoch in range(5):
            trainer.train(epoch=epoch)
            # score = trainer.test(epoch=epoch, run_on_finetune=True)
            # LOGGER.info("F2-Score: {}".format(score))
            # if score > best_score:
            #     is_best = True
            #     best_score = score
            # else:
            #     is_best = False
        save_checkpoint({'epoch': epoch + 1, 'f2_score': score,
                         'state_dict': trainer.model.state_dict()},
                        experiment_path,
                        backup_as_best=is_best)
        # score = trainer.test(epoch='final', save_submission=True)

    elif args.mode == 'finetune':
        LOGGER.info("Starting fine-tuning ...")
        trainer.lower_lr()
        for epoch in range(50):
            trainer.finetune(epoch=epoch)
        score = trainer.test(epoch='final', save_submission=True)

    elif args.mode == 'test':
        LOGGER.info("Starting inference on test set ...")
        score = trainer.test(epoch='final', save_submission=True)
