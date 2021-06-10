"""
Modified from https://github.com/seoungwugoh/STM/blob/master/dataset.py
"""

import os
from os import path
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data.dataset import Dataset
from dataset.range_transform import im_normalization
from dataset.util import all_to_onehot


class DAVISTestDataset(Dataset):
    def __init__(self, root, imset='2017/val.txt', resolution=480, single_object=False, target_name=None):
        self.root = root
        if resolution == 480:
            res_tag = '480p'
        else:
            res_tag = 'Full-Resolution'
        self.mask_dir = path.join(root, 'Annotations', res_tag)
        self.mask480_dir = path.join(root, 'Annotations', '480p')
        self.image_dir = path.join(root, 'JPEGImages', res_tag)
        self.resolution = resolution
        _imset_dir = path.join(root, 'ImageSets')
        _imset_f = path.join(_imset_dir, imset)

        self.videos = []
        self.num_frames = {}
        self.num_objects = {}
        self.shape = {}
        self.size_480p = {}
        with open(path.join(_imset_f), "r") as lines:
            for line in lines:
                _video = line.rstrip('\n')
                if target_name is not None and target_name != _video:
                    continue
                self.videos.append(_video)
                self.num_frames[_video] = len(os.listdir(path.join(self.image_dir, _video)))
                _mask = np.array(Image.open(path.join(self.mask_dir, _video, '00000.png')).convert("P"))
                self.num_objects[_video] = np.max(_mask)
                self.shape[_video] = np.shape(_mask)
                _mask480 = np.array(Image.open(path.join(self.mask480_dir, _video, '00000.png')).convert("P"))
                self.size_480p[_video] = np.shape(_mask480)

        self.single_object = single_object

        if resolution == 480:
            self.im_transform = transforms.Compose([
                transforms.ToTensor(),
                im_normalization,
            ])
        else:
            self.im_transform = transforms.Compose([
                transforms.ToTensor(),
                im_normalization,
                transforms.Resize(resolution, interpolation=InterpolationMode.BICUBIC),
            ])
            self.mask_transform = transforms.Compose([
                transforms.Resize(resolution, interpolation=InterpolationMode.NEAREST),
            ])

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video = self.videos[index]
        info = {}
        info['name'] = video
        info['frames'] = []
        info['num_frames'] = self.num_frames[video]
        info['size_480p'] = self.size_480p[video]

        images = []
        masks = []
        for f in range(self.num_frames[video]):
            img_file = path.join(self.image_dir, video, '{:05d}.jpg'.format(f))
            images.append(self.im_transform(Image.open(img_file).convert('RGB')))
            info['frames'].append('{:05d}.jpg'.format(f))
            
            mask_file = path.join(self.mask_dir, video, '{:05d}.png'.format(f))
            if path.exists(mask_file):
                masks.append(np.array(Image.open(mask_file).convert('P'), dtype=np.uint8))
            else:
                # Test-set maybe?
                masks.append(np.zeros_like(masks[0]))
        
        images = torch.stack(images, 0)
        masks = np.stack(masks, 0)
        
        if self.single_object:
            labels = [1]
            masks = (masks > 0.5).astype(np.uint8)
            masks = torch.from_numpy(all_to_onehot(masks, labels)).float()
        else:
            labels = np.unique(masks[0])
            labels = labels[labels!=0]
            masks = torch.from_numpy(all_to_onehot(masks, labels)).float()

        if self.resolution != 480:
            masks = self.mask_transform(masks)
        masks = masks.unsqueeze(2)

        info['labels'] = labels

        data = {
            'rgb': images,
            'gt': masks,
            'info': info,
        }

        return data

