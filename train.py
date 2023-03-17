from __future__ import absolute_import, division, print_function

import numpy as np
import time
import math
from options import LiteMonoOptions
from networks.depth_encoder import LiteMono
from networks.depth_decoder import DepthDecoder
from networks.pose_decoder import PoseDecoder
from networks.resnet_encoder import ResnetEncoder
import model_state_management
from training_logger import TrainingLogger
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed


class LiteMonoTrainer:

    def __init__(self, options: LiteMonoOptions) -> None:
        self.options = options

        # checking height and width are multiples of 32
        assert self.options.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.options.width % 32 == 0, "'width' must be a multiple of 32"
        assert self.options.frame_ids[0] == 0, "frame_ids must start with 0"


        self.log_path = os.path.join(self.options.log_dir, f"{self.options.model_name}-{math.floor(time.time())}")
        
        self.models = {}
        self.params_to_train = []

        self.device = torch.device("cpu" if self.options.no_cuda else "cuda")

        self.num_scales = len(self.options.scales)
        self.num_pose_frames = 2 if self.options.pose_model_input == "pairs" else len(self.options.frame_ids)

        self.__setup_lite_mono_model()
        self.__setup_pose_net_model()

        # Optimizer and model initialization
        self.model_optimizer = optim.Adam(self.params_to_train, self.options.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, self.options.scheduler_step_site, 0.1)

        if self.options.load_weights_folder is not None:
            model_state_management.load_model(self.options.load_weights_folder, self.options.models_to_load, self.models, self.model_optimizer)

        # Model Summary
        print("Training model named:\n  ", self.options.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.options.log_dir)
        print("Training is using:\n  ", self.device)

        self.__setup_dataset()

        # Dataset Summary
        print("Using split:\n  ", self.options.split)
        print("There are {:d} training items and {:d} validation items\n".format(len(self.train_loader.dataset), len(self.val_loader.dataset)))

        # Setup logging
        self.logger = TrainingLogger(self.options.batch_size, self.options.num_epochs, len(self.train_loader.dataset))

    def __setup_lite_mono_model(self):
        # setup for LiteMono architecture
        self.models["encoder"] = LiteMono()
        self.models["encoder"].to_device(self.device)
        self.params_to_train += list(self.models["encoder"].parameters())

        self.models["depth_decoder"] = DepthDecoder(self.models["encoder"].num_ch_enc, self.options.scales)
        self.models["depth_decoder"].to_device(self.device)
        self.params_to_train += list(self.models["depth_decoder"].parameters())

    def __setup_pose_net_model(self):
        # setup for PoseNet architecture
        self.models["pose_encoder"] = ResnetEncoder(self.options.num_layers, pretrained=self.options.weights_init == "pretrained", num_input_images=self.num_pose_frames)
        self.models["pose_encoder"].to_device(self.device)
        self.params_to_train += list(self.models["pose_encoder"].parameters())

        self.models["pose_decoder"] = PoseDecoder(self.models["pose_encoder"].num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)
        self.models["pose_decoder"].to_device(self.device)
        self.params_to_train += list(self.models["pose_decoder"].parameters())

    def __setup_dataset(self):
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.options.dataset]

        # selects a predefined data split from the ./splits folder 
        fpath = os.path.join(os.path.dirname(__file__), "splits", self.options.split, "{}_files.txt")
        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.options.png else '.jpg'
        num_train_samples = len(train_filenames)

        self.num_total_steps = num_train_samples // self.options.batch_size * self.options.num_epochs

        train_dataset = self.dataset(self.options.data_path, train_filenames, self.options.height, self.options.width, self.options.frame_ids, 4, is_train=True, img_ext=img_ext)
        # Select a subset of the training data
        train_dataset, _ = torch.utils.data.random_split(train_dataset, [self.options.data_percentage, 1.0 - self.options.data_percentage])
        self.train_loader = DataLoader(train_dataset, self.options.batch_size, True, num_workers=self.options.num_workers, pin_memory=True, drop_last=True)
        
        val_dataset = self.dataset( self.options.data_path, val_filenames, self.options.height, self.options.width, self.options.frame_ids, 4, is_train=False, img_ext=img_ext)
        # Select a subset of the validation data
        val_dataset, _ = torch.utils.data.random_split(val_dataset, [self.options.data_percentage, 1.0 - self.options.data_percentage])
        self.val_loader = DataLoader(val_dataset, self.options.batch_size, True, num_workers=self.options.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        if not self.options.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.options.scales:
            h = self.options.height // (2 ** scale)
            w = self.options.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.options.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.options.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        model_state_management.save_model_config(self.options, self.log_path)

    def __set_train_mode(self):
        for m in self.models.values():
            m.train()

    def __set_eval_mode(self):
        for m in self.models.values():
            m.eval()

    def train(self):
        self.current_epoch = 0
        self.current_step = 0
        self.logger.start()

        for self.current_epoch in range(self.options.num_epochs):
            self.__run_epoch()
            if (self.current_epoch + 1) % self.options.save_frequency == 0:
                model_state_management.save_model(self.log_path, f"weights_{self.current_epoch}", self.models, self.options, self.model_optimizer)

    def __run_epoch(self):
        self.model_lr_scheduler.step()
        
        self.__set_train_mode()

        for batch_idx, inputs in enumerate(self.train_loader):

            start = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - start

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.options.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                # TODO call Logger.log_time
                self.logger.log
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1


if __name__ == "__main__":
    options = LiteMonoOptions()
    opts = options.parse()

    trainer = LiteMonoTrainer(opts)
