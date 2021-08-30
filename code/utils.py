import datetime
import dateutil
import dateutil.tz
import os
import sys
import errno
import numpy as np

from config import cfg

# https://github.com/pytorch/examples/blob/master/imagenet/main.py
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def set_datadir(
    root_datadir,
    num_train,
    baseline_mode,
    curriculum_mode,
):
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    if cfg.HOLDOUT.MODE:
        holdout_mode = "holdoutmode_on"
    else:
        holdout_mode = "holdoutmode_off"
    datadir = os.path.join(
        root_datadir,
        "{}_episodes_{}_{}_{}_{}_trainingseed{}".format(
            now,
            num_train,
            baseline_mode,
            curriculum_mode,
            holdout_mode,
            cfg.TRAIN.SEED,
        ),
    )
    mkdir_p(datadir)
    print("Saving output to: {}".format(datadir))
    return datadir


class Logger(object):
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        sys.stdout.flush()
        self.log.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def get_category(obj_name):
    if obj_name in cfg.TRAIN.SMALL_OBJECTS:
        return 0
    elif obj_name in cfg.TRAIN.MEDIUM_OBJECTS:
        return 1
    elif obj_name in cfg.TRAIN.LARGE_OBJECTS:
        return 2


def generate_holdout_pairs():
    np.random.seed(cfg.HOLDOUT.SEED)  # For randomly selecting holdout pairs
    L_M_pairs = [
        [L, M] for L in cfg.TRAIN.LARGE_OBJECTS for M in cfg.TRAIN.MEDIUM_OBJECTS
    ]
    L_S_pairs = [
        [L, S] for L in cfg.TRAIN.LARGE_OBJECTS for S in cfg.TRAIN.SMALL_OBJECTS
    ]
    M_S_pairs = [
        [M, S] for M in cfg.TRAIN.MEDIUM_OBJECTS for S in cfg.TRAIN.SMALL_OBJECTS
    ]
    L_M_holdout_pairs = [
        L_M_pairs[i]
        for i in np.random.choice(
            len(L_M_pairs), size=cfg.HOLDOUT.PAIRS_PER_TYPE, replace=False
        )
    ]
    L_S_holdout_pairs = [
        L_S_pairs[i]
        for i in np.random.choice(
            len(L_S_pairs), size=cfg.HOLDOUT.PAIRS_PER_TYPE, replace=False
        )
    ]
    M_S_holdout_pairs = [
        M_S_pairs[i]
        for i in np.random.choice(
            len(M_S_pairs), size=cfg.HOLDOUT.PAIRS_PER_TYPE, replace=False
        )
    ]
    holdout_pairs = L_M_holdout_pairs + L_S_holdout_pairs + M_S_holdout_pairs
    return holdout_pairs


def generate_test_sample_holdout_pair():
    obj_set_idx = np.random.choice(len(cfg.HOLDOUT.PAIRS))
    obj_set = cfg.HOLDOUT.PAIRS[obj_set_idx]
    y_label = np.random.choice(["no", "yes"])
    if y_label == "yes":
        query_word = np.random.choice(obj_set)
        pass
    elif y_label == "no":
        query_wordset = list(set(cfg.TRAIN.ALL_OBJECTS) - set(obj_set))
        query_word = np.random.choice(query_wordset)
        pass
    else:
        raise
    y_label = 1 if y_label == "yes" else 0
    return query_word, obj_set, y_label


def generate_sample(obj_number, if_occlusion):
    # The query and existence prediction are generated randomly.
    # A object set is generated based on them then.
    query_word = np.random.choice(list(cfg.TRAIN.DICT.keys()))
    y_label = np.random.choice(["no", "yes"])  # Of equal probability
    occlusion = if_occlusion

    # Generate object set
    if obj_number == "one":
        if y_label == "no":
            obj_list = [obj for obj in cfg.TRAIN.ALL_OBJECTS if obj != query_word]
            obj_selected = np.random.choice(obj_list)
            obj_set = [obj_selected]
        else:
            obj_set = [query_word]

    elif obj_number == "two":
        if y_label == "no":
            if occlusion == "no":
                obj_list = [obj for obj in cfg.TRAIN.ALL_OBJECTS if obj != query_word]
                obj1, obj2 = np.random.choice(obj_list, size=2, replace=False)
                if cfg.HOLDOUT.MODE:
                    while [obj1, obj2] in cfg.HOLDOUT.PAIRS or [
                        obj2,
                        obj1,
                    ] in cfg.HOLDOUT.PAIRS:
                        obj1, obj2 = np.random.choice(obj_list, size=2, replace=False)
                obj_set = [obj1, obj2]
            elif occlusion == "yes":
                obj_list = [obj for obj in cfg.TRAIN.ALL_OBJECTS if obj != query_word]
                obj1, obj2 = np.random.choice(obj_list, size=2, replace=False)
                while get_category(obj1) == get_category(
                    obj2
                ):  # Make sure two objects are from different size categories.
                    obj1, obj2 = np.random.choice(obj_list, size=2, replace=False)
                if cfg.HOLDOUT.MODE:
                    while [obj1, obj2] in cfg.HOLDOUT.PAIRS or [
                        obj2,
                        obj1,
                    ] in cfg.HOLDOUT.PAIRS:
                        obj1, obj2 = np.random.choice(obj_list, size=2, replace=False)
                        while get_category(obj1) == get_category(obj2):
                            obj1, obj2 = np.random.choice(
                                obj_list, size=2, replace=False
                            )
                obj_set = [obj1, obj2]
        elif y_label == "yes":
            if occlusion == "no":
                obj1 = query_word
                obj_list = [obj for obj in cfg.TRAIN.ALL_OBJECTS if obj != obj1]
                obj2 = np.random.choice(obj_list)
                if cfg.HOLDOUT.MODE:
                    while [obj1, obj2] in cfg.HOLDOUT.PAIRS or [
                        obj2,
                        obj1,
                    ] in cfg.HOLDOUT.PAIRS:
                        obj2 = np.random.choice(obj_list)
                obj_set = [obj1, obj2]
            elif occlusion == "yes":
                obj1 = query_word
                if obj1 in cfg.TRAIN.LARGE_OBJECTS:
                    obj2 = np.random.choice(
                        cfg.TRAIN.MEDIUM_OBJECTS + cfg.TRAIN.SMALL_OBJECTS
                    )
                elif obj1 in cfg.TRAIN.MEDIUM_OBJECTS:
                    obj2 = np.random.choice(
                        cfg.TRAIN.LARGE_OBJECTS + cfg.TRAIN.SMALL_OBJECTS
                    )
                elif obj1 in cfg.TRAIN.SMALL_OBJECTS:
                    obj2 = np.random.choice(
                        cfg.TRAIN.LARGE_OBJECTS + cfg.TRAIN.MEDIUM_OBJECTS
                    )

                if cfg.HOLDOUT.MODE:
                    while [obj1, obj2] in cfg.HOLDOUT.PAIRS or [
                        obj2,
                        obj1,
                    ] in cfg.HOLDOUT.PAIRS:
                        if obj1 in cfg.TRAIN.LARGE_OBJECTS:
                            obj2 = np.random.choice(
                                cfg.TRAIN.MEDIUM_OBJECTS + cfg.TRAIN.SMALL_OBJECTS
                            )
                        elif obj1 in cfg.TRAIN.MEDIUM_OBJECTS:
                            obj2 = np.random.choice(
                                cfg.TRAIN.LARGE_OBJECTS + cfg.TRAIN.SMALL_OBJECTS
                            )
                        elif obj1 in cfg.TRAIN.SMALL_OBJECTS:
                            obj2 = np.random.choice(
                                cfg.TRAIN.LARGE_OBJECTS + cfg.TRAIN.MEDIUM_OBJECTS
                            )

                obj_set = [obj1, obj2]

    elif obj_number == "three":
        if y_label == "no":
            if occlusion == "no":
                obj_list = [obj for obj in cfg.TRAIN.ALL_OBJECTS if obj != query_word]
                obj1, obj2, obj3 = np.random.choice(obj_list, size=3, replace=False)
                obj_set = [obj1, obj2, obj3]
                pass
            elif (
                occlusion == "yes"
            ):  # three objs should be from at least two different size categories.
                obj_list = [obj for obj in cfg.TRAIN.ALL_OBJECTS if obj != query_word]
                # make sure obj1, obj2, obj3 are not from the same size category.
                obj1, obj2, obj3 = np.random.choice(obj_list, size=3, replace=False)
                while get_category(obj1) == get_category(obj2) == get_category(obj3):
                    obj1, obj2, obj3 = np.random.choice(obj_list, size=3, replace=False)

                obj_set = [obj1, obj2, obj3]
                pass
        elif y_label == "yes":
            if occlusion == "no":
                obj1 = query_word
                obj_list = [obj for obj in cfg.TRAIN.ALL_OBJECTS if obj != obj1]
                obj2, obj3 = np.random.choice(obj_list, size=2, replace=False)
                obj_set = [obj1, obj2, obj3]
                pass
            elif (
                occlusion == "yes"
            ):  # three objs should be from at least two different size categories.
                obj1 = query_word
                obj_list = [obj for obj in cfg.TRAIN.ALL_OBJECTS if obj != obj1]
                obj2, obj3 = np.random.choice(obj_list, size=2, replace=False)
                while get_category(obj1) == get_category(obj2) == get_category(obj3):
                    obj2, obj3 = np.random.choice(obj_list, size=2, replace=False)
                obj_set = [obj1, obj2, obj3]
                pass

    y_label = 1 if y_label == "yes" else 0
    return query_word, obj_set, y_label
