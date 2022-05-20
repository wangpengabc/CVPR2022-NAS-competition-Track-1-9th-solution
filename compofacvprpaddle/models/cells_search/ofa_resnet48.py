# CompOFA – Compound Once-For-All Networks for Faster Multi-Platform Deployment
# Under blind review at ICLR 2021: https://openreview.net/forum?id=IgIk8RRT-Z
#
# Implementation based on:
# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import copy
import random
import time

import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
# import paddle.DataParallel
import paddle.optimizer
import paddle.vision

from ..dynamic_layers import DynamicMBConvLayer, DynamicConvLayer, DynamicLinearLayer, DynamicResConvLayer
from ..basic_layers import MyNetwork, PoolingLayer, ConvLayer, IdentityLayer, LinearLayer, MBInvertedConvLayer
from ..layers_utils import make_divisible
from ...utils import int2list


class OFAResNet48(MyNetwork):
    """

    ks_list:
    初始化过程中， 格式为[3, 5, 7];
    set_active_subnet时，ks_list 格式为[3, [3,3], [3,3] ...];
    set_constraint时，格式为[3, 5, 7]

    expand_ratio_list:
    初始化过程中， 格式为[1.0, 0.8, 0.75];
    set_active_subnet时，格式为[0.9, [0.9,0.94], [0.8,0.85] ...];
    set_constraint时，格式为[1.0, 0.8, 0.75]

    depth_list:   --1,2,4 -- 3  -- stage
    初始化过程中， 格式为[[5,2], [8,2]]; 表示对应的stage最大depth
    set_active_subnet时，格式为[3, 5, 7, 2];
    set_constraint时，格式为[[5,2], [8,2]]

    """

    def __init__(self, n_classes=1000, bn_param=(0.1, 1e-5), dropout_rate=0.1, stem_conv_expand_ratio=1.0, base_stage_width=None,
                 width_mult_list=1.0, ks_list=3, expand_ratio_list=6, depth_list=4, compound=False, fixed_kernel=False):
        super(OFAResNet48, self).__init__()

        self.stem_conv_expand_ratio = stem_conv_expand_ratio
        self.width_mult_list = int2list(width_mult_list, 1)
        self.ks_list = int2list(ks_list, 1)
        self.ks_list = [3]
        # self.expand_ratio_list = int2list(expand_ratio_list, 1)
        # self.expand_ratio_list = [1.0, 0.95, 0.90, 0.85, 0.8, 0.75, 0.7]
        self.expand_ratio_list = expand_ratio_list
        # self.depth_list = int2list(depth_list, 1)

        depth_stage_124 = [i for i in range(depth_list[0][1], depth_list[0][0] + 1)]
        depth_stage_3 = [i for i in range(depth_list[1][1], depth_list[1][0] + 1)]
        self.depth_list = [depth_stage_124, depth_stage_3]
        # self.depth_list = [[2, 3, 4, 5], [2, 3, 4, 5, 6, 7, 8]]
        self.compound = compound
        self.fixed_kernel = fixed_kernel

        self.width_mult_list.sort()
        self.ks_list.sort()
        self.expand_ratio_list.sort()
        # self.depth_list.sort()

        base_stage_width = [64, 128, 256, 512]
        n_block_list = [5]*2 + [8]*1 + [5]*2

        width_list = []
        for base_width in base_stage_width:
            width = [make_divisible(base_width * width_mult, 8) for width_mult in self.width_mult_list]
            width_list.append(width)

        stride_stages = [1, 2, 2, 2]
        act_stages = ['relu', 'relu', 'relu', 'relu']

        # stem conv layer
        input_channel = make_divisible(base_stage_width[0] * max(self.expand_ratio_list), 8)
        # self.stem_block = nn.Sequential([DynamicConvLayer(
        #         in_channel_list=3, out_channel_list=input_channel, kernel_size=7,
        #         stride=2, act_func='relu',
        #     ),
        #     PoolingLayer(input_channel=input_channel, out_channels=input_channel, kernel_size=3, stride=2, act_func=None, pool_type='max')])

        self.stem_block = DynamicConvLayer(
            in_channel_list=3, out_channel_list=input_channel, kernel_size=7,
            stride=2, act_func='relu',
        )
        self.stem_max_pool_layer = PoolingLayer(in_channels=input_channel, out_channels=input_channel,
                                                kernel_size=3, stride=2, pool_type='max')

        # inverted residual blocks
        self.block_group_info = []
        self.blocks = []
        _block_index = 0
        feature_dim = [input_channel]

        for width, n_block, s, act_func in zip(width_list, n_block_list,
                                                       stride_stages, act_stages):
            self.block_group_info.append([_block_index + i for i in range(n_block)])
            _block_index += n_block

            output_channel = width
            for i in range(n_block):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                res_double_conv = DynamicResConvLayer(
                    in_channel_list=feature_dim, out_channel_list=output_channel, kernel_size_list=ks_list,
                    expand_ratio_list=expand_ratio_list, stride=stride, act_func=act_func
                )
                self.blocks.append(res_double_conv)
                feature_dim = output_channel
        self.blocks = nn.ModuleList(self.blocks)

        # self.final_expand_layer = ConvLayer(max(feature_dim), final_expand_width, kernel_size=1, act_func=None)
        self.classifier = DynamicLinearLayer(feature_dim, n_classes, dropout_rate=dropout_rate)

        # set bn param
        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

        # runtime_depth
        self.runtime_depth = [len(block_idx) for block_idx in self.block_group_info]

    """ MyNetwork required methods """

    @staticmethod
    def name():
        return 'OFAResNet48'

    def forward(self, x):
        # first block
        x = self.stem_block(x)
        x = self.stem_max_pool_layer(x)

        # blocks
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                # if hvd.rank() == 0:
                #     print("stage_id:{}, idx:{}".format(stage_id, idx))
                #     print(self.blocks[idx].module_str)
                x = self.blocks[idx](x)

        x = x.mean(3, keepdim=True).mean(2, keepdim=True)  # global average pooling
        x = paddle.squeeze(x)
        x = self.classifier(x)
        return x

    @property
    def module_str(self):
        _str = self.stem_block.module_str + '\n'
        _str += self.stem_max_pool_layer.module_str + '\n'

        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                _str += self.blocks[idx].module_str + '\n'

        # _str += self.feature_mix_layer.module_str + '\n'
        _str += self.classifier.module_str + '\n'
        return _str

    @property
    def config(self):
        return {
            'name': OFAResNet48.__name__,
            'bn': self.get_bn_param(),
            'first_conv': self.stem_block.config,
            'blocks': [
                block.config for block in self.blocks
            ],
            'classifier': self.classifier.config,
        }

    @staticmethod
    def build_from_config(config):
        raise ValueError('do not support this function')

    def load_weights_from_net(self, src_model_dict):
        model_dict = self.state_dict()
        for key in src_model_dict:
            if key in model_dict:
                new_key = key
            elif '.bn.bn.' in key:
                new_key = key.replace('.bn.bn.', '.bn.')
            elif '.conv.conv.weight' in key:
                new_key = key.replace('.conv.conv.weight', '.conv.weight')
            elif '.linear.linear.' in key:
                new_key = key.replace('.linear.linear.', '.linear.')
            ##############################################################################
            elif '.linear.' in key:
                new_key = key.replace('.linear.', '.linear.linear.')
            elif 'bn.' in key:
                new_key = key.replace('bn.', 'bn.bn.')
            elif 'conv.weight' in key:
                new_key = key.replace('conv.weight', 'conv.conv.weight')
            else:
                raise ValueError(key)
            assert new_key in model_dict, '%s' % new_key
            model_dict[new_key] = src_model_dict[key]
        self.load_state_dict(model_dict)

    """ set, sample and get active sub-networks """

    def set_active_subnet(self, wid=None, ks=None, e=None, d=None):
        # ks = [3, [3,3], [3,3] ..... ] for every block has two conv or int
        # e = [0.9, [0.9,0.94], [0.8,0.85] ...] for every block  or int
        # d = [5, 5, 8, 5]
        if self.fixed_kernel:
            assert ks is None, "You tried to set kernel size for a fixed kernel network!"
            # ks = [3, [3, 3], [3, 3], [3, 3], .....]
            ks = [3, ]
            for _ in range(len(self.blocks)):
                ks.append([3, 3])

        if isinstance(ks, int):
            ks = [ks, [ks, ks]*len(self.blocks)]
        elif isinstance(ks, list):
            ks = int2list(ks, len(self.blocks) + 1)
        # print(ks)

        if isinstance(e, int) or isinstance(e, float):
            expand_ratio = [e, [e, e]*len(self.blocks)]
        elif isinstance(e, list):
            expand_ratio = int2list(e, len(self.blocks) + 1)
        depth = int2list(d, len(self.block_group_info))

        # print(expand_ratio)
        # print(depth)

        # stem_block
        self.stem_block.expand_ratio = expand_ratio[0]

        # blocks
        for block, k, e in zip(self.blocks, ks[1:], expand_ratio[1:]):
            if k is not None:
                block.active_kernel_size = k
            if e is not None:
                block.active_expand_ratio = e

        for i, d in enumerate(depth):
            if d is not None:
                self.runtime_depth[i] = min(len(self.block_group_info[i]), d)

    def set_constraint(self, include_list, constraint_type='depth'):
        if constraint_type == 'depth':
            # self.__dict__['_depth_include_list'] = include_list.copy()
            depth_stage_124 = [i for i in range(include_list[0][1], include_list[0][0]+1)]
            depth_stage_3 = [i for i in range(include_list[1][1], include_list[1][0]+1)]
            self.__dict__['_depth_include_list'] = [depth_stage_124, depth_stage_3]
        elif constraint_type == 'expand_ratio':
            self.__dict__['_expand_include_list'] = include_list.copy()
        elif constraint_type == 'kernel_size':
            self.__dict__['_ks_include_list'] = include_list.copy()
        elif constraint_type == 'width_mult':
            self.__dict__['_widthMult_include_list'] = include_list.copy()
        else:
            raise NotImplementedError

    def clear_constraint(self):
        self.__dict__['_depth_include_list'] = None
        self.__dict__['_expand_include_list'] = None
        self.__dict__['_ks_include_list'] = None
        self.__dict__['_widthMult_include_list'] = None

    def sample_active_subnet(self):
        if self.compound:
            return self.sample_compound_subnet()

        ks_candidates = self.ks_list if self.__dict__.get('_ks_include_list', None) is None \
            else self.__dict__['_ks_include_list']
        expand_candidates = self.expand_ratio_list if self.__dict__.get('_expand_include_list', None) is None \
            else self.__dict__['_expand_include_list']
        depth_candidates = self.depth_list if self.__dict__.get('_depth_include_list', None) is None else \
            self.__dict__['_depth_include_list']

        # sample width_mult
        width_mult_setting = None
        random.seed(time.time())

        if self.fixed_kernel:
            ks_setting = None
        else:
            # sample kernel size
            # TODO
            pass

        # sample expand ratio
        expand_setting = []
        if not isinstance(expand_candidates[0], list):
            expand_candidates = [expand_candidates for _ in range(len(self.blocks)*2 + 1)]
        for e_set in expand_candidates:
            e = random.choice(e_set)
            expand_setting.append(e)
        # print(expand_setting)
        #split into [x, [x,x], [x,x], ....] format
        expand_setting_refactor = [expand_setting[0]]
        for i in range(len(self.blocks)):
            expand_setting_refactor.append(expand_setting[1+i*2:3+i*2])

        # sample depth
        depth_setting = []
        depth_candidates = [depth_candidates[0] if i != 2 else depth_candidates[1] for i in range(len(self.block_group_info))]
        for d_set in depth_candidates:
            d = random.choice(d_set)
            depth_setting.append(d)

        self.set_active_subnet(width_mult_setting, ks_setting, expand_setting_refactor, depth_setting)

        return {
            'wid': width_mult_setting,
            'ks': ks_setting,
            'e': expand_setting,
            'd': depth_setting,
        }

    def sample_compound_subnet(self):
        ks_candidates = self.ks_list if self.__dict__.get('_ks_include_list', None) is None \
            else self.__dict__['_ks_include_list']
        expand_candidates = self.expand_ratio_list if self.__dict__.get('_expand_include_list', None) is None \
            else self.__dict__['_expand_include_list']
        depth_candidates = self.depth_list if self.__dict__.get('_depth_include_list', None) is None else \
            self.__dict__['_depth_include_list']

        def clip_expands(expands):
            low = min(self.expand_ratio_list)
            expands = list(set(np.clip(expands, low, None)))
            return expands

        depth_candidates = self.depth_list
        mapping = {
            2: clip_expands([3, ]),
            3: clip_expands([4, ]),
            4: clip_expands([6, ]),
        }

        # used in in case of unbalanced distribution to sample proportional w/ cardinality
        combinations_per_depth = {d: len(mapping[d]) ** d for d in depth_candidates}
        sum_combinations = sum(combinations_per_depth.values())
        # print("sum_combinations", sum_combinations) # 3
        # print("combinations_per_depth", combinations_per_depth) # {2: 1, 3: 1, 4: 1}
        depth_sampling_weights = {k: v / sum_combinations for k, v in combinations_per_depth.items()}
        # print("depth_sampling_weights", depth_sampling_weights) # {2: 0.3333333333333333, 3: 0.3333333333333333, 4: 0.3333333333333333}

        width_mult_setting = None
        depth_setting = []
        expand_setting = []
        for block_idx in self.block_group_info:
            # for each block, sample a random depth weighted by the number of combinations
            # for each layer in block, sample from corresponding expand ratio
            sampled_d = np.random.choice(depth_candidates, p=list(depth_sampling_weights.values()))
            corresp_e = mapping[sampled_d]

            depth_setting.append(sampled_d)
            for _ in range(len(block_idx)):
                expand_setting.append(random.choice(corresp_e))

        if self.fixed_kernel:
            ks_setting = None
        else:
            # sample kernel size
            ks_setting = []
            if not isinstance(ks_candidates[0], list):
                ks_candidates = [ks_candidates for _ in range(len(self.blocks) - 1)]
            for k_set in ks_candidates:
                k = random.choice(k_set)
                ks_setting.append(k)

        self.set_active_subnet(width_mult_setting, ks_setting, expand_setting, depth_setting)

        # example:
        # {'wid': None, 'ks': [5, 7, 3, 5, 5, 5, 7, 7, 3, 7, 5, 5, 7, 5, 3, 7, 3, 3, 7, 5], ~ 20
        #  'e': [4, 4, 4, 4, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], ~ 20 = 4(max-depth) * 5(block) 'd': [3, 4, 3, 3, 3]}
        return {
            'wid': width_mult_setting,
            'ks': ks_setting,
            'e': expand_setting,
            'd': depth_setting,
        }

    # def get_active_subnet(self, preserve_weight=True):
    #     first_conv = copy.deepcopy(self.first_conv)
    #     blocks = [copy.deepcopy(self.blocks[0])]
    #
    #     final_expand_layer = copy.deepcopy(self.final_expand_layer)
    #     feature_mix_layer = copy.deepcopy(self.feature_mix_layer)
    #     classifier = copy.deepcopy(self.classifier)
    #
    #     input_channel = blocks[0].mobile_inverted_conv.out_channels
    #     # blocks
    #     for stage_id, block_idx in enumerate(self.block_group_info):
    #         depth = self.runtime_depth[stage_id]
    #         active_idx = block_idx[:depth]
    #         stage_blocks = []
    #         for idx in active_idx:
    #             stage_blocks.append(MobileInvertedResidualBlock(
    #                 self.blocks[idx].mobile_inverted_conv.get_active_subnet(input_channel, preserve_weight),
    #                 copy.deepcopy(self.blocks[idx].shortcut)
    #             ))
    #             input_channel = stage_blocks[-1].mobile_inverted_conv.out_channels
    #         blocks += stage_blocks
    #
    #     _subnet = MobileNetV3(first_conv, blocks, final_expand_layer, feature_mix_layer, classifier)
    #     _subnet.set_bn_param(**self.get_bn_param())
    #     return _subnet

    # def get_active_net_config(self):
    #     # first conv
    #     first_conv_config = self.first_conv.config
    #     first_block_config = self.blocks[0].config
    #     if isinstance(self.first_conv, DynamicConvLayer):
    #         first_conv_config = self.first_conv.get_active_subnet_config(3)
    #         first_block_config = {
    #             'name': MobileInvertedResidualBlock.__name__,
    #             'mobile_inverted_conv': self.blocks[0].mobile_inverted_conv.get_active_subnet_config(
    #                 first_conv_config['out_channels']
    #             ),
    #             'shortcut': self.blocks[0].shortcut.config if self.blocks[0].shortcut is not None else None,
    #         }
    #     final_expand_config = self.final_expand_layer.config
    #     feature_mix_layer_config = self.feature_mix_layer.config
    #     if isinstance(self.final_expand_layer, DynamicConvLayer):
    #         final_expand_config = self.final_expand_layer.get_active_subnet_config(
    #             self.blocks[-1].mobile_inverted_conv.active_out_channel)
    #         feature_mix_layer_config = self.feature_mix_layer.get_active_subnet_config(
    #             final_expand_config['out_channels'])
    #     classifier_config = self.classifier.config
    #     if isinstance(self.classifier, DynamicLinearLayer):
    #         classifier_config = self.classifier.get_active_subnet_config(self.feature_mix_layer.active_out_channel)
    #
    #     block_config_list = [first_block_config]
    #     input_channel = first_block_config['mobile_inverted_conv']['out_channels']
    #     for stage_id, block_idx in enumerate(self.block_group_info):
    #         depth = self.runtime_depth[stage_id]
    #         active_idx = block_idx[:depth]
    #         stage_blocks = []
    #         for idx in active_idx:
    #             middle_channel = make_divisible(round(input_channel *
    #                                                   self.blocks[idx].mobile_inverted_conv.active_expand_ratio), 8)
    #             stage_blocks.append({
    #                 'name': MobileInvertedResidualBlock.__name__,
    #                 'mobile_inverted_conv': {
    #                     'name': MBInvertedConvLayer.__name__,
    #                     'in_channels': input_channel,
    #                     'out_channels': self.blocks[idx].mobile_inverted_conv.active_out_channel,
    #                     'kernel_size': self.blocks[idx].mobile_inverted_conv.active_kernel_size,
    #                     'stride': self.blocks[idx].mobile_inverted_conv.stride,
    #                     'expand_ratio': self.blocks[idx].mobile_inverted_conv.active_expand_ratio,
    #                     'mid_channels': middle_channel,
    #                     'act_func': self.blocks[idx].mobile_inverted_conv.act_func,
    #                     'use_se': self.blocks[idx].mobile_inverted_conv.use_se,
    #                 },
    #                 'shortcut': self.blocks[idx].shortcut.config if self.blocks[idx].shortcut is not None else None,
    #             })
    #             input_channel = self.blocks[idx].mobile_inverted_conv.active_out_channel
    #         block_config_list += stage_blocks
    #
    #     return {
    #         'name': MobileNetV3.__name__,
    #         'bn': self.get_bn_param(),
    #         'first_conv': first_conv_config,
    #         'blocks': block_config_list,
    #         'final_expand_layer': final_expand_config,
    #         'feature_mix_layer': feature_mix_layer_config,
    #         'classifier': classifier_config,
    #     }

    """ Width Related Methods """

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        for block in self.blocks:
            block.re_organize_middle_weights(expand_ratio_stage)
