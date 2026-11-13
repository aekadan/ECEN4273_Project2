#############################################################
#   FILE NAME: test_train.py
#
#   DESCRIPTION: This file contains unit tests for the
#   training loop (Library/train.py) and the checkpoint
#   helpers (Library/model.py). Every test runs on the CPU
#   with pretrained=False, num_workers=0, and a few tiny
#   synthetic samples, so no GPU or weight download is needed.
#
#   NOTE: Everything below is commented out on purpose. It is
#   a plan of the tests to write, not the tests themselves.
#   Uncomment each piece as you implement it.
#
#   Authors: Group 2
#############################################################

#############################################################
#   Libraries
#############################################################

import json
import pytest
import torch

#############################################################
#   Files Import
#############################################################

from Library import train
from Library.dataset import IMAGE_SIZE, SyntheticTileDataset
from Library.model import save_checkpoint, load_checkpoint, create_model

#############################################################
#   Global Constants
#############################################################

TINY_TRAIN_SAMPLES = 16
TINY_VAL_SAMPLES = 8
TINY_BATCH_SIZE = 8

#############################################################
#   Test Classes
#############################################################

