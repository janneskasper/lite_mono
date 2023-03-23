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

from torch import default_generator, randperm
from torch._utils import _accumulate
from torch.utils.data.dataset import Subset


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
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, self.options.scheduler_step_size, 0.1)

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
        self.logger = TrainingLogger(self.log_path, self.options, len(self.train_loader.dataset))

    def __setup_lite_mono_model(self):
        # setup for LiteMono architecture
        self.models["encoder"] = LiteMono()
        self.models["encoder"].to(self.device)
        self.params_to_train += list(self.models["encoder"].parameters())

        self.models["depth_decoder"] = DepthDecoder(self.models["encoder"].num_ch_enc, self.options.scales)
        self.models["depth_decoder"].to(self.device)
        self.params_to_train += list(self.models["depth_decoder"].parameters())

    def __setup_pose_net_model(self):
        # setup for PoseNet architecture
        self.models["pose_encoder"] = ResnetEncoder(self.options.num_layers, pretrained=self.options.weights_init == "pretrained", num_input_images=self.num_pose_frames)
        self.models["pose_encoder"].to(self.device)
        self.params_to_train += list(self.models["pose_encoder"].parameters())

        self.models["pose_decoder"] = PoseDecoder(self.models["pose_encoder"].num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)
        self.models["pose_decoder"].to(self.device)
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
        train_dataset, _ = self.random_split(train_dataset, [self.options.data_percentage, 1.0 - self.options.data_percentage])
        self.train_loader = DataLoader(train_dataset, self.options.batch_size, True, num_workers=self.options.num_workers, pin_memory=True, drop_last=True)
        
        val_dataset = self.dataset( self.options.data_path, val_filenames, self.options.height, self.options.width, self.options.frame_ids, 4, is_train=False, img_ext=img_ext)
        # Select a subset of the validation data
        val_dataset, _ = self.random_split(val_dataset, [self.options.data_percentage, 1.0 - self.options.data_percentage])
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

    def __set_train_model(self):
        for m in self.models.values():
            m.train()

    def __set_eval_model(self):
        for m in self.models.values():
            m.eval()

    def train(self):
        self.current_epoch = 0
        self.current_step = 0
        self.logger.start()

        for self.current_epoch in range(self.options.num_epochs):
            print(f"Running epoch: {self.current_epoch}")
            self.__run_epoch()
            if (self.current_epoch + 1) % self.options.save_frequency == 0:
                model_state_management.save_model(self.log_path, f"weights_{self.current_epoch}", self.models, self.options, self.model_optimizer)

    def __run_epoch(self):
        
        print("Before training ...")
        self.__set_train_model()
        print("After training ...")

        for batch_idx, inputs in enumerate(self.train_loader):

            start = time.time()

            outputs, losses = self.__process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()
            self.model_lr_scheduler.step()

            duration = time.time() - start

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.options.log_frequency == 0 and self.current_step < 2000
            late_phase = self.current_step % 2000 == 0

            if early_phase or late_phase:
                self.logger.log_time(batch_idx, duration, losses["loss"].cpu().data, self.current_epoch, self.current_step)

                if "depth_gt" in inputs:
                    self.__compute_depth_losses(inputs, outputs, losses)

                self.logger.log("train", inputs, outputs, losses, self.current_step)
                self.__val()

            print(f"Step: {self.current_step}")
            self.current_step += 1

    def __process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        print(f"Moving data to {self.device}")
        for key, ipt in inputs.items():
            # print(".")
            inputs[key] = ipt.to(self.device)
        print(f"Moving data to {self.device} done !")

        features = self.models["encoder"](inputs["color_aug", 0, 0])
        outputs = self.models["depth_decoder"](features)

        if self.options.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        # Outputs is a list of dicts, which it shouldnt be
        outputs = outputs[0]
        outputs.update(self.__predict_poses(inputs, features))

        self.__generate_images_pred(inputs, outputs)
        losses = self.__compute_losses(inputs, outputs)

        return outputs, losses

    def __predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.options.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.options.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.options.frame_ids}

            for f_i in self.options.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    axisangle, translation = self.models["pose_decoder"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        return outputs

    def __generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.options.scales:
            disp = outputs[("disp", scale)]
            if self.options.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.options.height, self.options.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.options.min_depth, self.options.max_depth)

            outputs[("depth_decoder", 0, scale)] = depth

            for i, frame_id in enumerate(self.options.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                if not self.options.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def __compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.options.scales:
            loss = 0
            reprojection_losses = []

            if self.options.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.options.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.__compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.options.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.options.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.__compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.options.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.options.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.options.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.options.height, self.options.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.options.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.options.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape, device=self.device) * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.options.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.options.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def __compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.options.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss
    
    def __compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        # Here depth was changed to depth_decoder
        depth_pred = outputs[("depth_decoder", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def __val(self):
        """Validate the model on a single minibatch
        """
        self.__set_eval_model()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.__process_batch(inputs)

            if "depth_gt" in inputs:
                self.__compute_depth_losses(inputs, outputs, losses)
            
            self.logger.log("val", inputs, outputs, losses, self.current_step)
            del inputs, outputs, losses

        self.__set_train_model()

    def random_split(self, dataset, lengths, generator=default_generator):
        """
        Randomly split a dataset into non-overlapping new datasets of given lengths.

        If a list of fractions that sum up to 1 is given,
        the lengths will be computed automatically as
        floor(frac * len(dataset)) for each fraction provided.

        After computing the lengths, if there are any remainders, 1 count will be
        distributed in round-robin fashion to the lengths
        until there are no remainders left.

        Optionally fix the generator for reproducible results, e.g.:

        >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))
        >>> random_split(range(30), [0.3, 0.3, 0.4], generator=torch.Generator(
        ...   ).manual_seed(42))

        Args:
            dataset (Dataset): Dataset to be split
            lengths (sequence): lengths or fractions of splits to be produced
            generator (Generator): Generator used for the random permutation.
        """
        assert isinstance(lengths, (list, tuple))
        if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
            subset_lengths: List[int] = []
            for i, frac in enumerate(lengths):
                if frac < 0 or frac > 1:
                    raise ValueError(f"Fraction at index {i} is not between 0 and 1")
                n_items_in_split = int(
                    math.floor(len(dataset) * frac)  # type: ignore[arg-type]
                )
                subset_lengths.append(n_items_in_split)
            remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
            # add 1 to all the lengths in round-robin fashion until the remainder is 0
            for i in range(remainder):
                idx_to_add_at = i % len(subset_lengths)
                subset_lengths[idx_to_add_at] += 1
            lengths = subset_lengths
            for i, length in enumerate(lengths):
                if length == 0:
                    warnings.warn(f"Length of split at index {i} is 0. "
                                f"This might result in an empty dataset.")

        # Cannot verify that dataset is Sized
        if sum(lengths) != len(dataset):    # type: ignore[arg-type]
            raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

        indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[call-overload]
        return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]


if __name__ == "__main__":
    options = LiteMonoOptions()
    opts = options.parse()

    trainer = LiteMonoTrainer(opts)
    trainer.train()
