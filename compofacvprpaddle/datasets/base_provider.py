# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import numpy as np
import time
import random
import os
import math

import paddle
import paddle.vision.transforms as transforms
import paddle.vision.transforms.functional as F


class DataProvider:
    SUB_SEED = 937162211  # random seed for sampling subset
    VALID_SEED = 2147483647  # random seed for the validation set

    @staticmethod
    def name():
        """ Return name of the dataset """
        raise NotImplementedError

    @property
    def data_shape(self):
        """ Return shape as python list of one data entry """
        raise NotImplementedError

    @property
    def n_classes(self):
        """ Return `int` of num classes """
        raise NotImplementedError

    @property
    def save_path(self):
        """ local path to save the data """
        raise NotImplementedError

    @property
    def data_url(self):
        """ link to download the data """
        raise NotImplementedError

    @staticmethod
    def random_sample_valid_set(train_size, valid_size):
        assert train_size > valid_size

        paddle.seed(DataProvider.VALID_SEED)
        rand_indexes = paddle.randperm(train_size).tolist()

        valid_indexes = rand_indexes[:valid_size]
        train_indexes = rand_indexes[valid_size:]
        return train_indexes, valid_indexes

    @staticmethod
    def labels_to_one_hot(n_classes, labels):
        new_labels = np.zeros((labels.shape[0], n_classes), dtype=np.float32)
        new_labels[range(labels.shape[0]), labels] = np.ones(labels.shape)
        return new_labels


# class MyRandomResizedCrop(transforms.RandomResizedCrop):
#     ACTIVE_SIZE = 224
#     IMAGE_SIZE_LIST = [224]
#
#     CONTINUOUS = False
#     SYNC_DISTRIBUTED = False
#
#     EPOCH = 0
#
#     def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
#         super(MyRandomResizedCrop, self).__init__(size=self.ACTIVE_SIZE, scale=scale, ratio=ratio)
#
#         self.IMAGE_SIZE_LIST = size
#         self.scale = scale
#         self.ratio = ratio
#
#     def forward(self, img):
#         i, j, h, w = self.get_params(img, self.scale, self.ratio)
#         return F.resized_crop(
#             img, i, j, h, w, (MyRandomResizedCrop.ACTIVE_SIZE, MyRandomResizedCrop.ACTIVE_SIZE), self.interpolation
#         )
#
#     @staticmethod
#     def get_candidate_image_size():
#         if MyRandomResizedCrop.CONTINUOUS:
#             min_size = min(MyRandomResizedCrop.IMAGE_SIZE_LIST)
#             max_size = max(MyRandomResizedCrop.IMAGE_SIZE_LIST)
#             candidate_sizes = []
#             for i in range(min_size, max_size + 1):
#                 if i % 4 == 0:
#                     candidate_sizes.append(i)
#         else:
#             candidate_sizes = MyRandomResizedCrop.IMAGE_SIZE_LIST
#
#         relative_probs = None
#         return candidate_sizes, relative_probs
#
#     @staticmethod
#     def sample_image_size(batch_id):
#         if MyRandomResizedCrop.SYNC_DISTRIBUTED:
#             _seed = int('%d%.3d' % (batch_id, MyRandomResizedCrop.EPOCH))
#         else:
#             _seed = os.getpid() + time.time()
#         random.seed(_seed)
#         candidate_sizes, relative_probs = MyRandomResizedCrop.get_candidate_image_size()
#         MyRandomResizedCrop.ACTIVE_SIZE = random.choices(candidate_sizes, weights=relative_probs)[0]

