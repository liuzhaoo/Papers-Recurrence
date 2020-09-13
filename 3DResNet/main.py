from pathlib import Path
import json
import random
import os

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD,lr_scheduler
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.backends import cudnn
import torchvision

from opts import parse_opts


from model import (generate_model, load_pretrained_model, make_data_parallel,
                   get_fine_tuning_parameters)

from spatial_transforms import (Compose, Normalize, Resize, CenterCrop,
                                CornerCrop, MultiScaleCornerCrop,
                                RandomResizedCrop, RandomHorizontalFlip,
                                ToTensor, ScaleValue, ColorJitter,
                                PickFirstChannels)

from dataset import get_training_data, get_validation_data, get_inference_data
from util import Logger, worker_init_fn, get_lr
from training import train_epoch
from validation import val_epoch