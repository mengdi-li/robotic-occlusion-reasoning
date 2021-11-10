# Based on https://github.com/tohinz/pytorch-mac-network/blob/master/code/config.py

from __future__ import division
from __future__ import print_function

import datetime
import dateutil
import dateutil.tz

import os
import os.path as osp
from os.path import dirname, join, abspath
import numpy as np
from easydict import EasyDict as edict
import math

__C = edict()
cfg = __C

__C.GPU_ID = "1"
__C.CUDA = True
__C.ROOT_DATADIR = join(dirname(abspath(__file__)), os.pardir, "data")

__C.IS_TRAIN = True
__C.HEADLESS = True

### Baseline mode
__C.RANDOM_MOVE = False
__C.MOVE_LEFT_MAX = False
__C.STAY = False
assert (
    __C.RANDOM_MOVE + __C.MOVE_LEFT_MAX + __C.STAY == 0
    or __C.RANDOM_MOVE + __C.MOVE_LEFT_MAX + __C.STAY == 1
)

### Simulator options
__C.SIM = edict()
__C.SIM.SCENE_FILE = join(
    join(dirname(abspath(__file__)), os.pardir, "scene"),
    "pepper-around-table-with-objs.ttt",
)
__C.SIM.VISION_RESOLUTION = [256, 256]
__C.SIM.ROBOT_RADIUS = 1.0
__C.SIM.TABLE_RADIUS = 0.5
__C.SIM.OBJECT_TABLE_MARGIN = 0.08
__C.SIM.TABLE_HEIGHT = 0.72
__C.SIM.ROBOT_CAM_HEIGHT = 1.082
__C.SIM.ALL_OBJECTS = [
    ["box", "cleanser", "laptop", "pitcher", "plant", "wine", "stuffed_animal"],
    [
        "apple",
        "baseball",
        "foam_brick",
        "mug",
        "rubiks_cube",
        "meat_can",
        "master_chef_can",
    ],
    ["bolt", "dice", "key", "marble", "card", "battery", "button_battery"],
]  # Note: make sure in the sequence [[big objects], [middle objects], [small objects]]

### Question network params
__C.QUESTIONNET = edict()
__C.QUESTIONNET.DIM = 10

### Vision network params
__C.VISIONNET = edict()
__C.VISIONNET.DIM = 256

### RNN params
__C.RNN = edict()
__C.RNN.INPUT_SIZE = __C.QUESTIONNET.DIM + __C.VISIONNET.DIM
__C.RNN.HIDDEN_SIZE = 256

### Movement network params
__C.MOVEMENTNET = edict()
__C.MOVEMENTNET.OUTPUT_SIZE = 3
__C.MOVEMENTNET.MOVEMENTNET_MAP = {"0": "left", "1": "right", "2": "stop"}

### Existence prediction network params
__C.PREDICTIONNET = edict()
__C.PREDICTIONNET.OUTPUT_SIZE = 2
__C.PREDICTIONNET.ANSWER_MAP = {"0": "no", "1": "yes"}

### Training options
__C.TRAIN = edict()
__C.TRAIN.CKPT_DIR = ""
__C.TRAIN.CKPT_ITER = 0

### Four possible curriculum modes: "one_obj", "two_objs_without_occlusion", "two_objs_with_occlusion", "one_or_two_objs"
### These four curriculum modes correspond to the four difficulty levels L1-1-vis, L2-2-vis, L3-2-occ, L4-overall respectively in the paper.
__C.TRAIN.CURRICULUM_MODE = "one_obj"
# __C.TRAIN.CURRICULUM_MODE = "two_objs_without_occlusion"
# __C.TRAIN.CURRICULUM_MODE = "two_objs_with_occlusion"
# __C.TRAIN.CURRICULUM_MODE = "one_or_two_objs"

__C.TRAIN.RESET_MOVEMENT_AND_BASELINE_MODULE = False

__C.TRAIN.INIT_LR = 1e-4
__C.TRAIN.WEIGHT_LOSS_REINFORCE = 1e-2

__C.TRAIN.EXP_DATADIR = ""
__C.TRAIN.SEED = 123
__C.TRAIN.EPISODES = 150000
__C.TRAIN.VISUAL_FREQ = 1000
__C.TRAIN.SAVE_IMAGES = False
__C.TRAIN.CKPT_FREQ = (
    50000  # If set it to 0, only the finla checkpoint after training is saved
)
__C.TRAIN.STEP_RADIANS = math.pi / 6
__C.TRAIN.MAX_STEPS = 6
__C.TRAIN.LARGE_OBJECTS = [
    "box",
    "cleanser",
    "laptop",
    "pitcher",
    "plant",
    "wine",
    "stuffed_animal",
]
__C.TRAIN.MEDIUM_OBJECTS = [
    "apple",
    "baseball",
    "foam_brick",
    "mug",
    "rubiks_cube",
    "meat_can",
    "master_chef_can",
]
__C.TRAIN.SMALL_OBJECTS = [
    "bolt",
    "dice",
    "key",
    "marble",
    "card",
    "battery",
    "button_battery",
]
__C.TRAIN.ALL_OBJECTS = (
    __C.TRAIN.LARGE_OBJECTS + __C.TRAIN.MEDIUM_OBJECTS + __C.TRAIN.SMALL_OBJECTS
)

# Build a dictionay for all objects used for training
__C.TRAIN.DICT = {}
dict_idx = 0
for obj in __C.TRAIN.ALL_OBJECTS:
    __C.TRAIN.DICT[obj] = dict_idx
    dict_idx += 1

# Holdout settings
__C.HOLDOUT = edict()
__C.HOLDOUT.MODE = False  # Note: holdout settings are only for scenes with two objects
__C.HOLDOUT.SEED = 234
__C.HOLDOUT.PAIRS_PER_TYPE = 14
__C.HOLDOUT.PAIRS = []

### Test options
__C.TEST = edict()

# Two test modes: "test_holdoutpairs", "use_training_settings".
# "test_holdoutpairs": only scenes with holdout object pairs are used for testing
# "use_training_settings": only scenes used for training are used for testing

# Test on data of different distributions:
# When __C.HOLDOUT.MODE = True, there are 5 test modes:
# "one_obj", "two_objs_without_occlusion_training", "two_objs_with_occlusion_training",
# "two_objs_without_occlusion_holdout", "two_objs_with_occlusion_holdout"
# When __C.HOLDOUT.MODE = False, there are 4 test modes:
# "one_obj", "two_objs_without_occlusion", "two_objs_with_occlusion", "one_or_two_objs"
__C.TEST.TEST_MODE = "one_or_two_objs"
# __C.TEST.TEST_MODE = "use_training_settings" # or "test_holdoutpairs"

__C.TEST.CKPT_DIR = ""
__C.TEST.CKPT_ITER = 0

__C.TEST.SEED = 1234
__C.TEST.EPISODES = 10000
__C.TEST.VISUAL_FREQ = 1000
__C.TEST.MAX_STEPS = __C.TRAIN.MAX_STEPS
__C.TEST.LARGE_OBJECTS = [
    "box",
    "cleanser",
    "laptop",
    "pitcher",
    "plant",
    "wine",
    "stuffed_animal",
]
__C.TEST.MEDIUM_OBJECTS = [
    "apple",
    "baseball",
    "foam_brick",
    "mug",
    "rubiks_cube",
    "meat_can",
    "master_chef_can",
]
__C.TEST.SMALL_OBJECTS = [
    "bolt",
    "dice",
    "key",
    "marble",
    "card",
    "battery",
    "button_battery",
]
__C.TEST.ALL_OBJECTS = (
    __C.TEST.LARGE_OBJECTS + __C.TEST.MEDIUM_OBJECTS + __C.TEST.SMALL_OBJECTS
)


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if not k in b:
            raise KeyError("{} is not a valid config key".format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            elif isinstance(b[k], list):
                v = v.split(",")
                v = [int(_v) for _v in v]
            elif b[k] is None:
                if v == "None":
                    continue
                else:
                    v = v
            else:
                raise ValueError(
                    ("Type mismatch ({} vs. {}) " "for config key: {}").format(
                        type(b[k]), type(v), k
                    )
                )

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print("Error under config key: {}".format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""

    from yaml import load

    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader

    with open(filename, "r") as f:
        yaml_cfg = edict(load(f, Loader=Loader))

    _merge_a_into_b(yaml_cfg, __C)
