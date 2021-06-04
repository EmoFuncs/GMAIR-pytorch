import os
import h5py
import numpy as np
import cv2

import torch

from gmair.config import config as cfg

class SimpleScatteredMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, in_file, mode = 'train'):
        super().__init__()
        self.dataset = h5py.File(in_file, 'r')['{}/full'.format(mode)]
        self.episode = None

        # static_img = self.dataset[9, ...]
        # img_size = cfg.INPUT_IMAGE_SHAPE[-1]
        # self.static_img = cv2.resize(static_img, dsize=(img_size,img_size))

    def __getitem__(self, index):
        ret = []

        obs = self.dataset['image'][index, ...] # TODO index fixed to 0
        # obs = self.static_img
        # obs = np.zeros_like(obs)
        obs = obs[..., None]  # Add channel dimension
        # obs = cv2.cvtColor(obs, cv2.COLOR_GRAY2RGB)
        
        image = np.moveaxis(obs, -1, 0)  # move from (x, y, c) to (c, x, y)

        bbox_w_c = self.dataset['bbox_w_c'][index, ...]  # TODO index fixed to 0

        digit_count = self.dataset['digit_count'][index, ...]

        return image, bbox_w_c, digit_count

    def __len__(self):
        return self.dataset['image'].shape[0]

