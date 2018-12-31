#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import os
from easydict import EasyDict
import numpy as np


_C = EasyDict()
cfg = _C
# data augument config
_C.expand_prob = 0.5
_C.expand_max_ratio = 4
_C.hue_prob = 0.5
_C.hue_delta = 18
_C.contrast_prob = 0.5
_C.contrast_delta = 0.5
_C.saturation_prob = 0.5
_C.saturation_delta = 0.5
_C.brightness_prob = 0.5
_C.brightness_delta = 0.125
_C.data_anchor_sampling_prob = 0.5
_C.min_face_size = 20.0
_C.apply_distort = True
_C.apply_expand = False
_C.img_mean = np.array([104., 117., 123.])[:, np.newaxis, np.newaxis].astype(
    'float32')
_C.resize_width = 1024
_C.resize_height = 1024
_C.scale = 1 / 127.0
_C.anchor_sampling = False
_C.filter_min_face = True

# dataset config

_C.HOME = '/home/lj/data/'
_C.TRAIN_FILE = './data/train.txt'
_C.VAL_FILE = './data/val.txt'


# evalution config
_C.FDDB_DIR = '/home/lj/data/FDDB'
_C.WIDER_DIR = '/home/lj/data/WIDER'
_C.AFW_DIR = '/home/lj/data/AFW'
_C.PASCAL_DIR = '/home/lj/data/PASCAL_FACE'


# train config
_C.MAX_STEPS = 120000
_C.LR_STEPS = (80000,100000,120000)
_C.EPOCHES = 300

# anchor config
_C.FEATURE_MAPS = [32, 16, 8]
_C.STEPS = [32,64,128]
_C.DENSITY = [[-3, -1, 1, 3], [-2, 2], [0]]
_C.ASPECT_RATIOS = ((1, 2, 4), (1,), (1,))
_C.ANCHOR_SIZES = [32, 256, 512]
_C.VARIANCE = [0.1, 0.2]
_C.CLIP = False	

# loss config
_C.NUM_CLASSES = 2
_C.OVERLAP_THRESH = 0.35
_C.NEG_POS_RATION = 7

# detection config
_C.NMS_THRESH = 0.3
_C.NMS_TOP_K = 5000
_C.KEEP_TOP_K = 750
_C.CONF_THRESH = 0.05
