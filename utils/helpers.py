import os
import shutil
import torch
import numpy as np
from distutils.dir_util import copy_tree


def create_logger(log_filename, display=True):
    """Create a log file for the experiment"""
    f = open(log_filename, "a")
    counter = [0]

    def logger(text):
        if display:
            print(text)
        f.write(text + "\n")
        counter[0] += 1
        if counter[0] % 10 == 0:
            f.flush()
            os.fsync(f.fileno())

    return logger, f.close


def makedir(path):
    """If the directory does not exist, create it"""
    if not os.path.exists(path):
        os.makedirs(path)


def save_experiment(xp_dir, configuration):
    """Save files about the experiment"""
    makedir(xp_dir)
    shutil.copy(src="./main.py", dst=xp_dir)
    shutil.copy(src=configuration, dst=xp_dir)
    dirpath = os.path.join(os.getcwd() + "/models/")
    copy_tree(src=dirpath, dst=xp_dir + "models/")


def list_of_distances(X, Y):
    """Calculate a list of distances"""
    return torch.sum(
        (torch.unsqueeze(X, dim=2) - torch.unsqueeze(Y.t(), dim=0)) ** 2, dim=1
    )


def find_high_activation_crop(activation_map, percentile=95):
    """Locate high activation region"""
    threshold = np.percentile(activation_map, percentile)
    mask = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:, j]) > 0.5:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:, j]) > 0.5:
            upper_x = j
            break
    return lower_y, upper_y + 1, lower_x, upper_x + 1
