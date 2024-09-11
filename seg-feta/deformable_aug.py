# -*- coding:utf-8 -*-
import math
import os
import random
import re
from glob import glob

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torchio as tio
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

class Augmentation3d(nn.Module):
    """
        Augmentation 3D on GPUs.

        Args:
            aug_parameters : the parameters for augmentation, a dictionary value, for details:

            rot_range_x, rot_range_y, rot_range_z are the rotation range of different axes.
            scale_range_x, scale_range_y, scale_range_z are the scale range of different axes.
            The larger scale_range_x is set, the image smaller.
            shift_range_x, shift_range_y, shift_range_z are the translation range of different axes.

            contrast and gray_shift are the range used for linear gray value shift,
            the output gray value v_o = v_i * contrast + gray_shift, v_i is the original gray value,

            flip_x, flip_y, flip_z are the bool value of flip or not of each axes.

            elastic_alpha, smooth_num, field_size are values for elastic transform.
            The smaller elastic_alpha is set, the smoother image is outputted.
            If elastic_alpha is set as 0, only affine transform is performed.

            size_o is the output shape. The __crop__ method will crop the central area with this shape.

    """

    def __init__(self, aug_parameters, gpu):
        super(Augmentation3d, self).__init__()

        self.rot_range_x = aug_parameters["rot_range_x"]
        self.rot_range_y = aug_parameters["rot_range_y"]
        self.rot_range_z = aug_parameters["rot_range_z"]
        self.scale_range_x = aug_parameters["scale_range_x"]
        self.scale_range_y = aug_parameters["scale_range_y"]
        self.scale_range_z = aug_parameters["scale_range_z"]
        self.shift_range_x = aug_parameters["shift_range_x"]
        self.shift_range_y = aug_parameters["shift_range_y"]
        self.shift_range_z = aug_parameters["shift_range_z"]
        self.contrast = aug_parameters["contrast"]
        self.gray_shift = aug_parameters["gray_shift"]
        self.flip_x = aug_parameters["flip_x"]
        self.flip_y = aug_parameters["flip_y"]
        self.flip_z = aug_parameters["flip_z"]
        self.elastic_alpha = aug_parameters["elastic_alpha"]
        self.smooth_num = aug_parameters["smooth_num"]
        self.field_size = aug_parameters["field_size"]
        self.size_o = aug_parameters["size_o"]
        self.gpu = gpu

    def forward(self, vol_list, itp_mode_list, pad_mode_list):
        vol_aug_list = self.__data_aug__(vol_list, itp_mode_list, pad_mode_list)
        vol_aug_list = self.__crop__(vol_aug_list)
        return vol_aug_list

    def __crop__(self, vol_list):
        center = [vol_list[0].size(2) // 2, vol_list[0].size(3) // 2, vol_list[0].size(4) // 2]
        z_s = center[0] - self.size_o[0] // 2
        z_e = z_s + self.size_o[0]
        y_s = center[1] - self.size_o[1] // 2
        y_e = y_s + self.size_o[1]
        x_s = center[2] - self.size_o[2] // 2
        x_e = x_s + self.size_o[2]

        vol_crop_list = []
        for vol in vol_list:
            vol_crop = vol[:, :, z_s:z_e, y_s:y_e, x_s:x_e]
            vol_crop_list.append(vol_crop)

        return vol_crop_list

    def __angle_axis_to_rotation_matrix__(self, angle_axis):
        def _compute_rotation_matrix(angle_axis, theta2, eps=1e-6):
            # We want to be careful to only evaluate the square root if the
            # norm of the angle_axis vector is greater than zero. Otherwise
            # we get a division by zero.
            k_one = 1.0
            theta = torch.sqrt(theta2)
            wxyz = angle_axis / (theta + eps)
            wx, wy, wz = torch.chunk(wxyz, 3, dim=1)
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)

            r00 = cos_theta + wx * wx * (k_one - cos_theta)
            r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
            r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
            r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
            r11 = cos_theta + wy * wy * (k_one - cos_theta)
            r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
            r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
            r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
            r22 = cos_theta + wz * wz * (k_one - cos_theta)
            rotation_matrix = torch.cat(
                [r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1)
            return rotation_matrix.view(-1, 3, 3)

        def _compute_rotation_matrix_taylor(angle_axis):
            rx, ry, rz = torch.chunk(angle_axis, 3, dim=1)
            k_one = torch.ones_like(rx)
            rotation_matrix = torch.cat(
                [k_one, -rz, ry, rz, k_one, -rx, -ry, rx, k_one], dim=1)
            return rotation_matrix.view(-1, 3, 3)

        # stolen from ceres/rotation.h

        _angle_axis = torch.unsqueeze(angle_axis, dim=1)
        theta2 = torch.matmul(_angle_axis, _angle_axis.transpose(1, 2))
        theta2 = torch.squeeze(theta2, dim=1)

        # compute rotation matrices
        rotation_matrix_normal = _compute_rotation_matrix(angle_axis, theta2)
        rotation_matrix_taylor = _compute_rotation_matrix_taylor(angle_axis)

        # create mask to handle both cases
        eps = 1e-6
        mask = (theta2 > eps).view(-1, 1, 1).to(theta2.device)
        mask_pos = (mask).type_as(theta2)
        mask_neg = (mask == False).type_as(theta2)  # noqa

        # create output pose matrix
        batch_size = angle_axis.shape[0]
        rotation_matrix = torch.eye(4).to(angle_axis.device).type_as(angle_axis)
        rotation_matrix = rotation_matrix.view(1, 4, 4).repeat(batch_size, 1, 1)
        # fill output matrix with masked values
        rotation_matrix[..., :3, :3] = \
            mask_pos * rotation_matrix_normal + mask_neg * rotation_matrix_taylor
        return rotation_matrix  # Nx4x4

    def __affine_elastic_transform_3d_gpu__(self, vol_list, rot, scale, shift, mode_list, pad_mode_list,
                                            alpha=[2.0, 2.0, 2.0], smooth_num=4, win=[5, 5, 5],
                                            field_size=[11, 11, 11]):
        aff_matrix = self.__angle_axis_to_rotation_matrix__(rot)  # Nx4x4
        aff_matrix[:, 0, 3] = shift[:, 0]  # * data.size(4)
        aff_matrix[:, 1, 3] = shift[:, 1]  # * data.size(3)
        aff_matrix[:, 2, 3] = shift[:, 2]  # * data.size(2)
        # if scale:
        aff_matrix[:, 0, 0] *= scale[:, 0]
        aff_matrix[:, 1, 0] *= scale[:, 0]
        aff_matrix[:, 2, 0] *= scale[:, 0]
        aff_matrix[:, 0, 1] *= scale[:, 1]
        aff_matrix[:, 1, 1] *= scale[:, 1]
        aff_matrix[:, 2, 1] *= scale[:, 1]
        aff_matrix[:, 0, 2] *= scale[:, 2]
        aff_matrix[:, 1, 2] *= scale[:, 2]
        aff_matrix[:, 2, 2] *= scale[:, 2]

        aff_matrix = aff_matrix[:, 0:3, :]

        grid = torch.nn.functional.affine_grid(aff_matrix, vol_list[0].size())

        pad = [win[i] // 2 for i in range(3)]
        fs = field_size
        dz = torch.rand(1, 1, fs[0] + pad[0] * 2, fs[1] + pad[1] * 2, fs[2] + pad[2] * 2)
        dy = torch.rand(1, 1, fs[0] + pad[0] * 2, fs[1] + pad[1] * 2, fs[2] + pad[2] * 2)
        dx = torch.rand(1, 1, fs[0] + pad[0] * 2, fs[1] + pad[1] * 2, fs[2] + pad[2] * 2)
        dz = (dz - 0.5) * 2.0 * alpha[0]
        dy = (dy - 0.5) * 2.0 * alpha[1]
        dx = (dx - 0.5) * 2.0 * alpha[2]

        for _ in range(smooth_num):
            dz = self.__smooth_3d__(dz, win)
            dy = self.__smooth_3d__(dy, win)
            dx = self.__smooth_3d__(dx, win)

        dz = dz[:, :, pad[0]:pad[0] + fs[0], pad[1]:pad[1] + fs[1], pad[2]:pad[2] + fs[2]]
        dy = dy[:, :, pad[0]:pad[0] + fs[0], pad[1]:pad[1] + fs[1], pad[2]:pad[2] + fs[2]]
        dx = dx[:, :, pad[0]:pad[0] + fs[0], pad[1]:pad[1] + fs[1], pad[2]:pad[2] + fs[2]]

        size_3d = [vol_list[0].size(2), vol_list[0].size(3), vol_list[0].size(4)]
        batch_size = vol_list[0].size(0)
        dz = self.__resize__(dz, size_3d).repeat(batch_size, 1, 1, 1, 1)
        dy = self.__resize__(dy, size_3d).repeat(batch_size, 1, 1, 1, 1)
        dx = self.__resize__(dx, size_3d).repeat(batch_size, 1, 1, 1, 1)

        grid[:, :, :, :, 0] += dz[:, 0, :, :, :]
        grid[:, :, :, :, 1] += dy[:, 0, :, :, :]
        grid[:, :, :, :, 2] += dx[:, 0, :, :, :]

        vol_o_list = []
        for i, vol in enumerate(vol_list):
            vol_o = torch.nn.functional.grid_sample(vol, grid, mode_list[i], pad_mode_list[i])
            vol_o_list.append(vol_o)
        return vol_o_list

    def __smooth_3d__(self, vol, win):
        kernel = torch.ones([1, vol.size(1), win[0], win[1], win[2]])
        pad_size = [(int)((win[2] - 1) / 2), (int)((win[2] - 1) / 2),
                    (int)((win[1] - 1) / 2), (int)((win[1] - 1) / 2),
                    (int)((win[0] - 1) / 2), (int)((win[0] - 1) / 2)]
        vol = torch.nn.functional.pad(vol, pad_size, "replicate")
        vol_s = torch.nn.functional.conv3d(vol, kernel, stride=(1, 1, 1)) / torch.sum(kernel)
        return vol_s

    def __resize__(self, vol, size_tgt, mode="trilinear"):
        vol_t = nn.functional.interpolate(vol, size=size_tgt, mode=mode, align_corners=False)

        return vol_t

    def __data_aug__(self, vol_list, itp_mode_list, pad_mode_list):
        N = vol_list[0].size(0)
        rand_rot_x = (torch.rand(N, 1) * (self.rot_range_x[1] - self.rot_range_x[0]) + self.rot_range_x[0]) \
                     / 180 * math.pi
        rand_rot_y = (torch.rand(N, 1) * (self.rot_range_y[1] - self.rot_range_y[0]) + self.rot_range_y[0]) \
                     / 180 * math.pi
        rand_rot_z = (torch.rand(N, 1) * (self.rot_range_z[1] - self.rot_range_z[0]) + self.rot_range_z[0]) \
                     / 180 * math.pi
        rand_rot = torch.cat([rand_rot_x, rand_rot_y, rand_rot_z], dim=1)

        rand_scale_x = torch.rand(N, 1) * (self.scale_range_x[1] - self.scale_range_x[0]) + self.scale_range_x[0]
        rand_scale_y = torch.rand(N, 1) * (self.scale_range_y[1] - self.scale_range_y[0]) + self.scale_range_y[0]
        rand_scale_z = torch.rand(N, 1) * (self.scale_range_z[1] - self.scale_range_z[0]) + self.scale_range_z[0]
        rand_scale = torch.cat([rand_scale_x, rand_scale_y, rand_scale_z], dim=1)

        rand_shift_x = torch.rand(N, 1) * (self.shift_range_x[1] - self.shift_range_x[0]) + self.shift_range_x[0]
        rand_shift_y = torch.rand(N, 1) * (self.shift_range_y[1] - self.shift_range_y[0]) + self.shift_range_y[0]
        rand_shift_z = torch.rand(N, 1) * (self.shift_range_z[1] - self.shift_range_z[0]) + self.shift_range_z[0]
        rand_shift = torch.cat([rand_shift_x, rand_shift_y, rand_shift_z], dim=1)

        # mode_list = ["nearest"] * len(vol_list)
        vol_aug_list = self.__affine_elastic_transform_3d_gpu__(
            vol_list,
            rand_rot, rand_scale, rand_shift,
            itp_mode_list, pad_mode_list,
            alpha=self.elastic_alpha, smooth_num=self.smooth_num, field_size=self.field_size)

        # flip
        if self.flip_x and random.randint(0, 1) == 0:
            for i, vol_aug in enumerate(vol_aug_list):
                vol_aug_list[i] = torch.flip(vol_aug_list[i], dims=[4])

        if self.flip_y and random.randint(0, 1) == 0:
            for i, vol_aug in enumerate(vol_aug_list):
                vol_aug_list[i] = torch.flip(vol_aug_list[i], dims=[3])

        if self.flip_z and random.randint(0, 1) == 0:
            for i, vol_aug in enumerate(vol_aug_list):
                vol_aug_list[i] = torch.flip(vol_aug_list[i], dims=[2])

        return vol_aug_list


def augmentation(mask, vol, aug_model, gpu):
    mask_torch = mask.float().unsqueeze(0).unsqueeze(0)
    vol_torch = vol.float().unsqueeze(0).unsqueeze(0)
    itp_mode_list = ["nearest", "bilinear"]
    pad_mode_list = ["zeros", "zeros"]
    mask_aug_torch = aug_model([mask_torch, vol_torch], itp_mode_list, pad_mode_list)
    mask_aug = mask_aug_torch[0][0, 0].numpy().astype("uint8")
    vol_aug = mask_aug_torch[1][0, 0].numpy().astype("float")
    return mask_aug, vol_aug


def aug_mask_and_img(mask, img):
    aug_parameters = {
        "rot_range_x": (-8.0, 8.0),
        "rot_range_y": (-8.0, 8.0),
        "rot_range_z": (-8.0, 8.0),
        "scale_range_x": (0.95, 1.00),
        "scale_range_y": (0.95, 1.00),
        "scale_range_z": (0.95, 1.00),
        "shift_range_x": (-0.02, 0.02),
        "shift_range_y": (-0.02, 0.02),
        "shift_range_z": (-0.02, 0.02),
        "contrast": (1.0, 1.0),
        "gray_shift": (0.0, 0.0),
        "flip_x": False,
        "flip_y": False,
        "flip_z": False,
        "elastic_alpha": [3.0, 3.0, 3.0],  # x,y,z
        "smooth_num": 4,
        "field_size": [10, 10, 10],  # x,y,z
        "size_o": (128, 128, 128)
    }
    aug_model = Augmentation3d(aug_parameters, gpu=0).cuda()
    gt_mask, hu_volume = augmentation(mask, img, aug_model, gpu=0)
    torch.cuda.empty_cache()
    return gt_mask, hu_volume
