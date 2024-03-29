"""Utility functions."""

# Imports.
import os, csv, logging, shutil
import torch
import numpy as np


# Log format.
LOG_FORMAT = '%(asctime)-15s %(levelname)-5s %(name)-15s - %(message)s'


def wrap_cuda(pytorch_obj):
    """Convert tensor object to CudaTensor.

    Loads on GPU if CUDA available.
    Allows for testing on CPU if CUDA not available.

    Args:
        pytorch_obj (torch.Tensor): torch tensor.

    Returns:
        torch.Tensor: Same tensor, Cuda type if CUDA available.

    """
    if torch.cuda.is_available():
        pytorch_obj = pytorch_obj.cuda()

    return pytorch_obj


def custom_collate(batch):
    """Create custom collate fn to check input size.

    Args:
        batch (list): list of items in a batch from data loader.

    Returns:
        (tuple): inputs, labels and image IDs.

    """
    inputs, labels, image_ids = [], [], []
    for item in batch:
        if item[0].size(0) == 3:
            inputs.append(np.expand_dims(item[0], 0))
            labels.append(np.expand_dims(item[1], 0))
            image_ids.append(item[2])
    inputs = torch.from_numpy(np.concatenate(inputs)).float()
    labels = torch.from_numpy(np.concatenate(labels)).float()

    return inputs, labels, image_ids


def compute_f_score(probs, label, threshold=0.25, beta=2):
    """Compute f-score.

    Args:
        probs (torch.Tensor): output probabilities.
        label (torch.Tensor): label tensor.
        threshold (float, optional): confidence threshold for a prediction.
        beta (int, optional): beta in F-beta score.

    Returns:
        (float): f-score.

    """
    probs = probs > threshold
    label = label > threshold

    TP = (probs & label).sum(1).float()
    TN = ((~probs) & (~label)).sum(1).float()
    FP = (probs & (~label)).sum(1).float()
    FN = ((~probs) & label).sum(1).float()

    precision = TP / (TP + FP + 1e-12)
    recall = TP / (TP + FN + 1e-12)
    score = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)

    return score.mean(0)


def save_checkpoint(state, target_dir, file_name='checkpoint.pth.tar',
                    backup_as_best=False,):
    """Save checkpoint to disk.

    Args:
        state: object to save
        target_dir (str): Full path to the directory in which the checkpoint
            will be stored
        backup_as_best (bool): Should we backup the checkpoint as the best
            version
        file_name (str): the name of the checkpoint

    """
    best_model_path = os.path.join(target_dir, 'model_best.pth.tar')
    target_model_path = os.path.join(target_dir, file_name)

    os.makedirs(target_dir, exist_ok=True)
    torch.save(state, target_model_path)
    if backup_as_best:
        shutil.copyfile(target_model_path, best_model_path)


def setup_logging(log_path=None, debug=False, logger=None, fmt=LOG_FORMAT):
    """Prepare logging for the provided logger.

    Args:
        log_path (str, optional): full path to the desired log file
        debug (bool, optional): log in verbose mode or not
        logger (logging.Logger, optional): logger to setup logging upon,
            if it's None, root logger will be used
        fmt (str, optional): format for the logging message

    """
    lvl = logging.DEBUG if debug else logging.INFO
    logger = logger if logger else logging.getLogger()
    logger.setLevel(lvl)
    logger.handlers = []

    fmt = logging.Formatter(fmt=fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if log_path:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
        logger.info('Log file is %s', log_path)


def read_csv(file_path):
    """Read csv file.

    Args:
        file_path (str): path to file.

    Returns:
        (list): list of contents read from csv file.

    """
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        out_list = list(reader)

        return out_list
