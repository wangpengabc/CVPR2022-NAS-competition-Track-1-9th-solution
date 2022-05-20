# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import warnings
import os
import math
import numpy as np

# import torch.utils.data
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets

import paddle.io
import paddle.vision.transforms as transforms
from paddle.vision.transforms import (
    RandomHorizontalFlip, RandomResizedCrop, SaturationTransform,
    Compose, Resize, HueTransform, BrightnessTransform, ContrastTransform,
    RandomCrop, Normalize, RandomRotation, CenterCrop)
from paddle.io import DataLoader


# from .base_provider import DataProvider, MyRandomResizedCrop, MyDistributedSampler
# from .base_provider import DataProvider, MyRandomResizedCrop
from .base_provider import DataProvider


class ImagenetDataProvider(DataProvider):
    # DEFAULT_PATH = None
    DEFAULT_PATH = "/home/sdb/yufang2/experiment/data/ILSVRC2012"

    def __init__(self, save_path=DEFAULT_PATH, train_batch_size=256, test_batch_size=512, valid_size=None, n_worker=32,
                 resize_scale=0.08, distort_color=None, image_size=224,
                 num_replicas=None, rank=None):
        # assert self.DEFAULT_PATH is None, "Set ImagenetDataProvider.DEFAULT_PATH"

        warnings.filterwarnings('ignore')
        self._save_path = save_path

        self.image_size = image_size  # int or list of int
        self.distort_color = distort_color
        self.resize_scale = resize_scale

        self._valid_transform_dict = {}
        # 默认条件下 image_size为int，不是list
        # if not isinstance(self.image_size, int):
        #     # TODO for variable input size
        #     assert isinstance(self.image_size, list)
        #     from .my_data_loader import MyDataLoader
        #     self.image_size.sort()  # e.g., 160 -> 224
        #     # MyRandomResizedCrop.IMAGE_SIZE_LIST = self.image_size.copy()
        #     # MyRandomResizedCrop.ACTIVE_SIZE = max(self.image_size)
        #
        #     for img_size in self.image_size:
        #         self._valid_transform_dict[img_size] = self.build_valid_transform(img_size)
        #     self.active_img_size = max(self.image_size)
        #     valid_transforms = self._valid_transform_dict[self.active_img_size]
        #     train_loader_class = MyDataLoader  # randomly sample image size for each batch of training image
        # else:
        self.active_img_size = self.image_size
        valid_transforms = self.build_valid_transform()
        train_loader_class = paddle.io.DataLoader

        train_transforms = self.build_train_transform()
        train_dataset = self.train_dataset(train_transforms)

        self.train = train_loader_class(
            train_dataset, batch_size=train_batch_size, shuffle=True,
            num_workers=n_worker)
        self.valid = None

        test_dataset = self.test_dataset(valid_transforms)
        self.test = paddle.io.DataLoader(
            test_dataset, batch_size=test_batch_size, shuffle=True, num_workers=n_worker)

        if self.valid is None:
            self.valid = self.test

    @staticmethod
    def name():
        return 'imagenet'

    @property
    def data_shape(self):
        return 3, self.active_img_size, self.active_img_size  # C, H, W

    @property
    def n_classes(self):
        return 1000

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = self.DEFAULT_PATH
        return self._save_path

    @property
    def data_url(self):
        raise ValueError('unable to download %s' % self.name())

    def train_dataset(self, _transforms):
        dataset = paddle.vision.datasets.ImageFolder(self.train_path, _transforms)
        return dataset

    def test_dataset(self, _transforms):
        dataset = paddle.vision.datasets.ImageFolder(self.valid_path, _transforms)
        return dataset

    @property
    def train_path(self):
        return os.path.join(self.save_path, 'train')

    @property
    def valid_path(self):
        return os.path.join(self.save_path, 'val')

    @property
    def normalize(self):
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def build_train_transform(self, image_size=None, print_log=True):
        if image_size is None:
            image_size = self.image_size
        if print_log:
            print('Color jitter: %s, resize_scale: %s, img_size: %s' %
                  (self.distort_color, self.resize_scale, image_size))

        resize_transform_class = transforms.RandomResizedCrop
        print("image_size", image_size)

        train_transforms = [
            resize_transform_class(image_size, scale=(self.resize_scale, 1.0)),
            transforms.RandomHorizontalFlip(),
        ]
        train_transforms += [
            ToArray(),
            self.normalize,
        ]
        train_transforms = transforms.Compose(train_transforms)
        return train_transforms

    def build_valid_transform(self, image_size=None):
        if image_size is None:
            image_size = self.active_img_size
        return Compose([
            transforms.Resize(int(math.ceil(image_size / 0.875))),
            transforms.CenterCrop(image_size),
            ToArray(),
            self.normalize
        ])

    def assign_active_img_size(self, new_img_size):
        self.active_img_size = new_img_size
        if self.active_img_size not in self._valid_transform_dict:
            self._valid_transform_dict[self.active_img_size] = self.build_valid_transform()
        # change the transform of the valid and test set
        self.valid.dataset.transform = self._valid_transform_dict[self.active_img_size]
        self.test.dataset.transform = self._valid_transform_dict[self.active_img_size]


class ToArray(object):
    def __call__(self, img):
        img = np.array(img)
        img = np.transpose(img, [2, 0, 1])
        img = img / 255.
        return img.astype('float32')