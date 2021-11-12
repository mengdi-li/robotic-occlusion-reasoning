import os
from os.path import dirname, join, abspath
import sys

import datetime
import dateutil
import dateutil.tz

import numpy as np
import random
import torch
import torch.nn.functional as F
import time
from tqdm import tqdm
import shutil

from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

import argparse
from config import cfg, cfg_from_file

from model import Model
from utils import (
    AverageMeter,
    set_datadir,
    mkdir_p,
    Logger,
    generate_holdout_pairs,
    generate_sample,
    generate_test_sample_holdout_pair,
)
from env import Env

from torch.utils.tensorboard import SummaryWriter

from PIL import Image

import torchvision
from torchvision import transforms

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_ID


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg_file", dest="cfg_file", help="optional config file", default="", type=str
    )
    args = parser.parse_args()
    return args

args = parse_args()
if args.cfg_file is not "":
    cfg_from_file(args.cfg_file)


class Trainer:
    def __init__(self, rank, gpu_id):
        if cfg.GPU_ID is not "" and torch.cuda.is_available():
            self.device = torch.device("cuda:%d" % gpu_id)
        else:
            self.device = torch.device("cpu")

        self.rank = rank
        self.gpu_id = gpu_id

        self.num_train = cfg.TRAIN.EPISODES
        self.max_steps = cfg.TRAIN.MAX_STEPS

        self.init_resnet()

        self.env = Env()
        self.env.pr.step()
        self.model = Model(
            len(cfg.TRAIN.DICT),
            cfg.QUESTIONNET.DIM,
            cfg.VISIONNET.DIM,
            cfg.RNN.INPUT_SIZE,
            cfg.RNN.HIDDEN_SIZE,
            cfg.MOVEMENTNET.OUTPUT_SIZE,
            cfg.PREDICTIONNET.OUTPUT_SIZE,
            self.device,
        )

        self.model.to(self.device)

        self.ddp_model = DDP(self.model, device_ids=[self.gpu_id]).to(self.device)  # Note: we only use one GPU here. # int(cfg.GPU_ID)

        self.optimizer = torch.optim.Adam(self.ddp_model.parameters(), lr=cfg.TRAIN.INIT_LR)

    def reset(self):
        h_t = torch.zeros(
            1,  # batch_size == 1
            cfg.RNN.HIDDEN_SIZE,
            dtype=torch.float,
            device=self.device,
            requires_grad=True,
        )
        return h_t

    def test(self):
        self.num_test = cfg.TEST.EPISODES
        print("\n[*] Test on {} episodes.".format(self.num_test))

        if cfg.TEST.CKPT_DIR != "":
            ckpt_file = "ckpt_{:08}.pth.tar".format(cfg.TEST.CKPT_ITER)
            test_checkpoint = join(
                cfg.ROOT_DATADIR,
                cfg.TEST.CKPT_DIR,
                "Model",
                ckpt_file,
            )
            self.load_check_point(test_checkpoint, self.gpu_id)
        else:
            raise Exception("No checkpoint for test!")

        # print holdout info
        if cfg.HOLDOUT.MODE:
            print("Holdout mode is on")
            print("Holdout pairs:{}".format(cfg.HOLDOUT.PAIRS))

        # logs
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        test_logfile = os.path.join(
            dirname(abspath(test_checkpoint)),
            os.pardir,
            "Log",
            "test_{}_{}_{}.log".format(now, cfg.TEST.TEST_MODE, cfg.TEST.CKPT_ITER),
        )
        sys.stdout = Logger(logfile=test_logfile)

        accs = AverageMeter()
        steps = AverageMeter()
        rewards = AverageMeter()

        max_steps = cfg.TEST.MAX_STEPS

        self.ddp_model.eval()

        pre_obj_set = []

        if cfg.HOLDOUT.MODE:
            assert cfg.TEST.TEST_MODE in [
                "two_objs_without_occlusion_holdout",
                "two_objs_with_occlusion_holdout",
                "one_obj",
                "two_objs_without_occlusion_training",
                "two_objs_with_occlusion_training",
            ]
        else:
            assert cfg.TEST.TEST_MODE in [
                "one_obj",
                "two_objs_without_occlusion",
                "two_objs_with_occlusion",
                "one_or_two_objs",
            ]

        with tqdm(total=self.num_test) as pbar:
            for episode in range(self.num_test):
                tic = time.time()
                if cfg.TEST.TEST_MODE == "two_objs_without_occlusion_holdout":
                    (
                        query_word,
                        obj_set,
                        y_label,
                    ) = generate_test_sample_holdout_pair()
                    obj_number = "two"
                    if_occlusion = "no"
                elif cfg.TEST.TEST_MODE == "two_objs_with_occlusion_holdout":
                    (
                        query_word,
                        obj_set,
                        y_label,
                    ) = generate_test_sample_holdout_pair()
                    obj_number = "two"
                    if_occlusion = "yes"
                else:
                    if cfg.TEST.TEST_MODE == "one_obj":
                        obj_number = "one"
                        if_occlusion = ""
                    elif (
                        cfg.TEST.TEST_MODE == "two_objs_without_occlusion"
                        or cfg.TEST.TEST_MODE == "two_objs_without_occlusion_training"
                    ):
                        obj_number = "two"
                        if_occlusion = "no"
                    elif (
                        cfg.TEST.TEST_MODE == "two_objs_with_occlusion"
                        or cfg.TEST.TEST_MODE == "two_objs_with_occlusion_training"
                    ):
                        obj_number = "two"
                        if_occlusion = "yes"
                    elif cfg.TEST.TEST_MODE == "one_or_two_objs":
                        obj_number = np.random.choice(["one", "two"], p=[0.333, 0.667])
                        if_occlusion = np.random.choice(["no", "yes"])

                    (
                        query_word,
                        obj_set,
                        y_label,
                    ) = generate_sample(obj_number, if_occlusion)

                y_label = torch.tensor(y_label).to(self.device)
                query_word_idx = torch.LongTensor([cfg.TRAIN.DICT[query_word]]).to(
                    self.device
                )
                # reset the env
                self.env.reset(
                    pre_obj_set,
                    obj_set,
                    obj_number,
                    if_occlusion,
                )
                pre_obj_set = obj_set

                h_t = self.reset()
                for t in range(max_steps):
                    movement_map = cfg.MOVEMENTNET.MOVEMENTNET_MAP
                    rgb_image = self.env.get_rgbimage()
                    rgb_image = np.uint8(rgb_image * 256.0)

                    rgb_image = (
                        torch.from_numpy(rgb_image).float().to(self.device)
                    )  # shape:[64,64,3]

                    image_feature = self.extract_image_feature(rgb_image)

                    h_t, m_t, b_t = self.ddp_model(query_word_idx, image_feature, h_t)

                    movement = torch.max(m_t, 1)[1]

                    if cfg.RANDOM_MOVE == True:
                        movement = np.random.choice([0, 1, 2])
                        movement = torch.tensor(movement)
                    elif cfg.MOVE_LEFT_MAX == True:
                        movement = 0
                        movement = torch.tensor(movement)
                    elif cfg.STAY == True:
                        movement = 2
                        movement = torch.tensor(movement)
                    else:
                        pass

                    if movement.item() != 2:
                        self.env.agent_circling(movement_map[str(movement.item())])
                    else:
                        break

                p_t = self.ddp_model(query_word_idx, image_feature, h_t, last=True)

                predicted_existence = torch.max(p_t, 1)[1]
                correct = (predicted_existence.detach() == y_label).float().item()

                if predicted_existence.detach() == y_label:
                    R_acc = 1
                else:
                    R_acc = -1

                R_lat = 1 / (t + 2)
                R = R_acc + R_lat

                accs.update(correct, 1)
                steps.update(t, 1)
                rewards.update(R, 1)

                toc = time.time()
                if episode != 0 and episode % cfg.TEST.VISUAL_FREQ == 0:
                    # calculate mean loss and acc
                    pbar.set_description(
                        (
                            "episode time:{:.1f}s - acc:{:.3f} - steps:{:.3f} - - Reward: {:.3f}".format(
                                (toc - tic), accs.avg, steps.avg, rewards.avg
                            )
                        )
                    )
                    pbar.update(cfg.TEST.VISUAL_FREQ)

        print(
            "\n[*]Test Acc: {} - Steps: {} - Reward: {}".format(
                accs.avg, steps.avg, rewards.avg
            )
        )

    def train(self, cfg_file):
        print("\n[*] Train on {} episodes.".format(self.num_train))

        if cfg.RANDOM_MOVE == True:
            baseline_mode = "randommove"
        elif cfg.MOVE_LEFT_MAX == True:
            baseline_mode = "moveleftmax"
        elif cfg.STAY == True:
            baseline_mode = "stay"
        else:
            baseline_mode = "nobaselinemode"

        if self.rank == 0:
            root_datadir = cfg.ROOT_DATADIR
            if cfg.TRAIN.EXP_DATADIR == "":
                data_dir = set_datadir(
                    root_datadir, self.num_train, baseline_mode, cfg.TRAIN.CURRICULUM_MODE
                )
            else:
                data_dir = os.path.join(root_datadir, cfg.TRAIN.EXP_DATADIR)
                mkdir_p(data_dir)
                print("Saving output to: {}".format(data_dir))

            self.model_dir = os.path.join(data_dir, "Model")
            self.log_dir = os.path.join(data_dir, "Log")
            self.code_dir = os.path.join(data_dir, "Code")
            self.sample_dir = os.path.join(data_dir, "Sample")
            self.cfg_dir = os.path.join(data_dir, "Config")
            mkdir_p(self.model_dir)
            mkdir_p(self.log_dir)
            mkdir_p(self.code_dir)
            mkdir_p(self.sample_dir)
            mkdir_p(self.cfg_dir)

            # log file
            train_logfile = os.path.join(self.log_dir, "train.log")
            sys.stdout = Logger(logfile=train_logfile)

            # print holdout info
            if cfg.HOLDOUT.MODE:
                print("Holdout mode is on.")
                print("Holdout pairs:{}".format(cfg.HOLDOUT.PAIRS))

            # copy code
            for filename in os.listdir(join(dirname(abspath(__file__)))):
                if filename.endswith(".py"):
                    shutil.copy(join(dirname(abspath(__file__)), filename), self.code_dir)

            # copy config yml
            shutil.copy(cfg_file, self.cfg_dir)

            self.writer = SummaryWriter(log_dir=self.log_dir)

        if cfg.TRAIN.CKPT_DIR != "":
            ckpt_file = "ckpt_{:08}.pth.tar".format(cfg.TRAIN.CKPT_ITER)
            train_checkpoint = join(
                cfg.ROOT_DATADIR,
                cfg.TRAIN.CKPT_DIR,
                "Model",
                ckpt_file,
            )
            self.load_check_point(train_checkpoint, self.gpu_id)

        self.ddp_model.train()

        if self.rank == 0:
            # for logs
            losses = AverageMeter()
            losses_prediction = AverageMeter()
            losses_baseline = AverageMeter()
            losses_reinforce = AverageMeter()
            accs = AverageMeter()
            steps = AverageMeter()
            adjusted_rewards = AverageMeter()
            baselines = AverageMeter()
            Rs = AverageMeter()
            # for each curriculum mode
            accs_one_obj = AverageMeter()
            steps_one_obj = AverageMeter()
            accs_two_objs_without_occlusion = AverageMeter()
            steps_two_objs_without_occlusion = AverageMeter()
            accs_two_objs_with_occlusion = AverageMeter()
            steps_two_objs_with_occlusion = AverageMeter()

        pre_obj_set = []  # for resetting a scene

        with tqdm(total=self.num_train) as pbar:
            for episode in range(self.num_train):
                tic = time.time()

                # set the object number and the occlusion situation according to a specified curriculum mode
                if cfg.TRAIN.CURRICULUM_MODE == "one_obj":
                    obj_number = "one"
                    if_occlusion = ""
                elif cfg.TRAIN.CURRICULUM_MODE == "two_objs_without_occlusion":
                    obj_number = "two"
                    if_occlusion = "no"
                elif cfg.TRAIN.CURRICULUM_MODE == "two_objs_with_occlusion":
                    obj_number = "two"
                    if_occlusion = "yes"
                elif cfg.TRAIN.CURRICULUM_MODE == "two_obj_with_or_without_occlusion":
                    obj_number = "two"
                    if_occlusion = np.random.choice(["no", "yes"])
                elif cfg.TRAIN.CURRICULUM_MODE == "one_or_two_objs":
                    obj_number = np.random.choice(["one", "two"], p=[0.333, 0.667])
                    if_occlusion = np.random.choice(["no", "yes"])
                else:
                    raise

                (
                    query_word,
                    obj_set,
                    y_label,
                ) = generate_sample(obj_number, if_occlusion)

                y_label = torch.tensor(y_label).to(self.device)

                # reset the env
                self.env.reset(
                    pre_obj_set,
                    obj_set,
                    obj_number,
                    if_occlusion,
                )
                pre_obj_set = obj_set

                (
                    loss,
                    loss_prediction,
                    loss_baseline,
                    loss_reinforce,
                    correct,
                    step,
                    adjusted_reward,
                    baseline,
                    R,
                ) = self.train_one_episode(query_word, y_label, episode)

                if self.rank == 0:
                    losses.update(loss, 1)
                    losses_prediction.update(loss_prediction, 1)
                    losses_baseline.update(loss_baseline, 1)
                    losses_reinforce.update(loss_reinforce, 1)
                    accs.update(correct, 1)
                    steps.update(step, 1)
                    adjusted_rewards.update(adjusted_reward, 1)
                    baselines.update(baseline, 1)
                    Rs.update(R, 1)

                    # for each curriculum mode
                    if obj_number == "one":
                        accs_one_obj.update(correct, 1)
                        steps_one_obj.update(step, 1)
                    elif obj_number == "two":
                        if if_occlusion == "no":
                            accs_two_objs_without_occlusion.update(correct, 1)
                            steps_two_objs_without_occlusion.update(step, 1)
                        elif if_occlusion == "yes":
                            accs_two_objs_with_occlusion.update(correct, 1)
                            steps_two_objs_with_occlusion.update(step, 1)
                        else:
                            raise
                    else:
                        raise

                    toc = time.time()
                    if episode != 0 and episode % cfg.TRAIN.VISUAL_FREQ == 0:
                        # calculate mean loss and acc
                        pbar.set_description(
                            (
                                "episode time:{:.1f}s - loss:{:.3f} - acc:{:.3f} - steps:{:.3f}".format(
                                    (toc - tic), losses.avg, accs.avg, steps.avg
                                )
                            )
                        )
                        pbar.update(cfg.TRAIN.VISUAL_FREQ)
                        self.writer.add_scalar("avg_loss", losses.avg, episode)
                        self.writer.add_scalar(
                            "avg_loss_prediction", losses_prediction.avg, episode
                        )
                        self.writer.add_scalar(
                            "avg_loss_baseline", losses_baseline.avg, episode
                        )
                        self.writer.add_scalar(
                            "avg_losses_reinforce", losses_reinforce.avg, episode
                        )
                        self.writer.add_scalar("avg_train_accuracy", accs.avg, episode)
                        self.writer.add_scalar("avg_steps", steps.avg, episode)
                        self.writer.add_scalar(
                            "adjusted_rewards", adjusted_rewards.avg, episode
                        )
                        self.writer.add_scalar("Rewards", Rs.avg, episode)

                        self.writer.add_scalar("accs_one_obj", accs_one_obj.avg, episode)
                        self.writer.add_scalar("steps_one_obj", steps_one_obj.avg, episode)
                        self.writer.add_scalar(
                            "accs_two_objs_without_occlusion",
                            accs_two_objs_without_occlusion.avg,
                            episode,
                        )
                        self.writer.add_scalar(
                            "steps_two_objs_without_occlusion",
                            steps_two_objs_without_occlusion.avg,
                            episode,
                        )
                        self.writer.add_scalar(
                            "accs_two_objs_with_occlusion",
                            accs_two_objs_with_occlusion.avg,
                            episode,
                        )
                        self.writer.add_scalar(
                            "steps_two_objs_with_occlusion",
                            steps_two_objs_with_occlusion.avg,
                            episode,
                        )

                        losses.reset()
                        losses_prediction.reset()
                        losses_baseline.reset()
                        losses_reinforce.reset()
                        accs.reset()
                        steps.reset()
                        adjusted_rewards.reset()
                        Rs.reset()
                        baselines.reset()
                        accs_one_obj.reset()
                        steps_one_obj.reset()
                        accs_two_objs_without_occlusion.reset()
                        steps_two_objs_without_occlusion.reset()
                        accs_two_objs_with_occlusion.reset()
                        steps_two_objs_with_occlusion.reset()

                    if (
                        cfg.TRAIN.CKPT_FREQ != 0
                        and episode != 0
                        and episode % cfg.TRAIN.CKPT_FREQ == 0
                    ):
                        self.save_checkpoint(
                            {
                                "episode": episode,
                                "model_state": self.ddp_model.state_dict(),
                                "optim_state": self.optimizer.state_dict(),
                            },
                            episode,
                        )

        if self.rank == 0:
            self.writer.close()
            self.save_checkpoint(
                {
                    "episode": self.num_train,
                    "model_state": self.ddp_model.state_dict(),
                    "optim_state": self.optimizer.state_dict(),
                },
                self.num_train,
            )

    def init_resnet(self):
        # preprocess
        self.preprocess_img_for_resnet = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # resnet
        self.resnet = torchvision.models.resnet18(pretrained=True)
        layers = [
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
        ]
        for i in range(2):  # use the first 2 layers of resnet50
            name = "layer%d" % (i + 1)
            layers.append(getattr(self.resnet, name))
        self.resnet = torch.nn.Sequential(*layers)
        for p in self.resnet.parameters():
            p.requires_grad = False
        self.resnet.to(self.device)
        self.resnet.eval()

    def extract_image_feature(self, rgb_image):
        # preprocess image
        rgb_image = rgb_image.unsqueeze(0)  # shape:[1,64,64,3], batch_size=1
        rgb_image = rgb_image.permute(0, 3, 1, 2)  # shape:[1,3,64,64]

        # get image features
        image_prep = np.zeros([len(rgb_image), len(rgb_image[0]), 224, 224], dtype=np.float32)
        image_prep = torch.from_numpy(image_prep).to(self.device)

        for i in range(len(rgb_image)):
            img = rgb_image[i]
            img_prep = self.preprocess_img_for_resnet(img)
            image_prep[i] = img_prep
        feature_batch = self.resnet(image_prep)
        return feature_batch

    def train_one_episode(self, query_word, y_label, episode):
        h_t = self.reset()
        rgb_image = None

        log_pi = []
        baselines = []

        predicted_existence = torch.tensor(-1).to(self.device)
        query_word_idx = torch.LongTensor([cfg.TRAIN.DICT[query_word]]).to(self.device)

        self.optimizer.zero_grad()

        for t in range(self.max_steps):
            # forward pass through model
            movement_map = cfg.MOVEMENTNET.MOVEMENTNET_MAP
            rgb_image = self.env.get_rgbimage()
            rgb_image = rgb_image * 255

            # save images for visualization
            if self.rank == 0:
                if cfg.TRAIN.SAVE_IMAGES and episode % cfg.TRAIN.VISUAL_FREQ == 0:
                    img = Image.fromarray(rgb_image, "RGB")
                    img.save(os.path.join(self.sample_dir, "%d-%d.jpg" % (episode, t)))

            rgb_image = (
                torch.from_numpy(rgb_image).float().to(self.device)
            )  # shape:[64,64,3]

            image_feature = self.extract_image_feature(rgb_image)

            h_t, m_t, b_t = self.ddp_model(query_word_idx, image_feature, h_t)

            baselines.append(b_t)

            m = torch.distributions.categorical.Categorical(m_t)
            movement = m.sample()
            log_pi_movement = m.log_prob(movement).squeeze()
            log_pi.append(log_pi_movement)

            # action selection strategies of baseline models
            if cfg.RANDOM_MOVE == True:
                movement = np.random.choice([0, 1, 2])
                movement = torch.tensor(movement)
            elif cfg.MOVE_LEFT_MAX == True:
                movement = 0
                movement = torch.tensor(movement)
            elif cfg.STAY == True:
                movement = 2
                movement = torch.tensor(movement)

            if movement.item() != 2:
                self.env.agent_circling(movement_map[str(movement.item())])
            else:
                break

        p_t = self.ddp_model(
            query_word_idx, image_feature, h_t, last=True
        )  # whether use the previous h_t?

        predicted_existence = torch.max(p_t, 1)[1]
        if predicted_existence.detach() == y_label:
            R_acc = 1
        else:
            R_acc = -1

        R_lat = 1 / (t + 2)
        R1 = R_acc + R_lat
        R1 = torch.tensor(R1).to(self.device)
        R = R1.repeat(t + 1)

        baselines = torch.stack(baselines).to(self.device)
        log_pi = torch.stack(log_pi).to(self.device)

        adjusted_reward = R - baselines.detach()

        loss_prediction = F.nll_loss(p_t, y_label.unsqueeze(0))
        loss_reinforce = torch.sum(-log_pi * adjusted_reward)
        loss_reinforce = torch.mean(loss_reinforce, dim=0)
        loss_baseline = F.mse_loss(baselines[: t + 1], R)

        if cfg.RANDOM_MOVE == True or cfg.MOVE_LEFT_MAX == True or cfg.STAY == True:
            loss_reinforce = torch.tensor(0, requires_grad=False)
            loss_baseline = torch.tensor(0, requires_grad=False)

        loss = (
            loss_prediction
            + loss_baseline
            + cfg.TRAIN.WEIGHT_LOSS_REINFORCE * loss_reinforce
        )

        # compute gradients and update parameters
        loss.backward()
        self.optimizer.step()

        correct = (predicted_existence.detach() == y_label).float()

        return (
            loss.item(),
            loss_prediction.item(),
            loss_baseline.item(),
            loss_reinforce.item(),
            correct.item(),
            t,
            adjusted_reward.mean().item(),
            baselines.mean().item(),
            R1.item(),
        )

    def save_checkpoint(self, state, iter):
        torch.save(state, "{}/ckpt_{:08}.pth.tar".format(self.model_dir, iter))

    def load_check_point(self, check_point, gpu_id):
        if os.path.isfile(check_point):
            print("=> loading checkpoint '{}'".format(check_point))
            map_location = {
                "cuda:%d" % 0: "cuda:%d" % gpu_id
            }  # assume the checkpoint is saved by cuda:0
            cp = torch.load(check_point, map_location)
            self.ddp_model.load_state_dict(cp["model_state"])
            self.optimizer.load_state_dict(cp["optim_state"])
            print("=> loaded checkpoint '{}'".format(check_point))

            ## debug
            if cfg.TRAIN.RESET_MOVEMENT_AND_BASELINE_MODULE:
                print("reset the movement_net and the baseliner module")
                self.ddp_model.movement_net.fc.reset_parameters()
                self.ddp_model.movement_net.fc_lt.reset_parameters()
                self.ddp_model.baseliner.fc.reset_parameters()

            """
            for layer in self.model.movement_net.children():
                # if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
            for layer in self.model.baseliner.children():
                # if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
            """

        else:
            print("=> no checkpoint found at '{}'".format(check_point))
            raise Exception("No checkpoint found!")

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size) # initialize the process group

def main_train(rank, world_size, args):
    setup(rank, world_size)
    random.seed(cfg.TRAIN.SEED + int(rank))
    np.random.seed(cfg.TRAIN.SEED + int(rank))
    torch.manual_seed(cfg.TRAIN.SEED + int(rank))
    # gpu_id = 0  # Note: here 0 means the first gpu in os.environ["CUDA_VISIBLE_DEVICES"]; We only use one gpu
    gpu_id = rank # Each process uses one gpu
    trainer = Trainer(rank, gpu_id)
    trainer.train(args.cfg_file)
    print("Done!")
    trainer.env.shutdown()

if __name__ == "__main__":

    if cfg.HOLDOUT.MODE:
        cfg.HOLDOUT.PAIRS = generate_holdout_pairs()
    else:
        cfg.HOLDOUT.PAIRS = []

    if cfg.IS_TRAIN:
        world_size = 4
        mp.spawn(main_train, args=(world_size,args), nprocs=world_size, join=True)
    else:
        random.seed(cfg.TEST.SEED)
        np.random.seed(cfg.TEST.SEED)
        torch.manual_seed(cfg.TEST.SEED)
        trainer = Trainer()
        trainer.test()

    print("Done!")
