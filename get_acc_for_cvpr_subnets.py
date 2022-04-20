"""

Pytorch distributed reference: https://github.com/tczhangzhi/pytorch-distributed

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python get_acc_for_cvpr_subnets.py
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python get_acc_for_cvpr_subnets.py
"""


import argparse
import json
import os
import random
import time
import numpy as np

import torch
import torch.nn as nn
import horovod.torch as hvd

from compofacvpr.datasets.imagenet import ImagenetDataProvider
from compofacvpr.procedures import ImagenetRunConfig, RunManager
from compofacvpr.datasets.base_provider import MyRandomResizedCrop
from compofacvpr.models.cells_search.ofa_resnet48 import OFAResNet48
from compofacvpr.procedures import DistributedImageNetRunConfig
from compofacvpr.procedures.distributed_run_manager import DistributedRunManager
from compofacvpr.datasets.base_provider import MyRandomResizedCrop
from compofacvpr.utils import download_url
from compofacvpr.procedures.progressive_shrinking import load_models

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_num", type=str, default="single", choices=["single", "multiple"])
parser.add_argument("--arch_idx_label", type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6, 7])
parser.add_argument("--arch_idx_gpu_split", type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6, 7])
parser.add_argument("--arch_idx_start", type=int, default=0)
parser.add_argument("--arch_idx_stop", type=int, default=0)

args = parser.parse_args()

if args.gpu_num == "multiple":
    split_idx = (args.arch_idx_label-1)*4 + args.arch_idx_gpu_split
    args.arch_idx_start = 2250 * split_idx
    args.arch_idx_stop = 2250 * (split_idx + 1)
    if args.arch_idx_stop > 45000:
        args.arch_idx_stop = 45000
elif args.gpu_num == "single":
    split_idx = args.arch_idx_gpu_split
    # args.arch_idx_start = 6500 * split_idx
    # args.arch_idx_stop = 6500 * (split_idx + 1)
    # if args.arch_idx_stop > 45000:
    #     args.arch_idx_stop = 45000
    args.arch_idx_start = 1300 * split_idx + 6500*3
    args.arch_idx_stop = 1300 * (split_idx + 1) + 6500*3
    if args.arch_idx_stop > 45000:
        args.arch_idx_stop = 45000

args.teacher_path = "/home/sdc/wangpeng/NASHello/CVPR2022Track1/runs/default/depth2depth_width/phase2/checkpoint/checkpoint.pth.tar"

args.manual_seed = 0

args.lr_schedule_type = 'cosine'

args.base_batch_size = 192
args.valid_size = None

args.opt_type = 'sgd'
args.momentum = 0.9
args.no_nesterov = False
args.weight_decay = 3e-5
args.label_smoothing = 0.1
args.no_decay_keys = 'bn#bias'
args.fp16_allreduce = False

args.model_init = 'he_fout'
args.validation_frequency = 1
args.print_frequency = 10

args.n_worker = 1
args.resize_scale = 0.08
args.distort_color = 'tf'
# args.image_size = '128,160,192,224'
args.image_size = '224'
args.continuous_size = True
args.not_sync_distributed_image_size = False

args.bn_momentum = 0.1
args.bn_eps = 1e-5
args.dropout = 0.1
args.base_stage_width = 'proxyless'

args.width_mult_list = '1.0'
args.dy_conv_scaling_mode = -1
args.independent_distributed_sampling = False

args.kd_ratio = 1.0
# args.kd_ratio = 0
args.kd_type = 'ce'

args.path = f'runs/eval'
args.dynamic_batch_size = 4
args.n_epochs = 120
args.base_lr = 0.24 * args.base_batch_size / 2048
args.warmup_epochs = 5
args.warmup_lr = -1
args.expand_list = '1.0,0.95,0.9,0.85,0.8,0.75,0.7'
args.depth_list = '5,2,8,2'
args.ks_list = '3'

args.distributed = False

gpus = [1, 2, 3, 4, 5, 6, 7]

def set_subnet(config):
    ks = None
    wid = None
    # extract config
    e_map = {'1':1.0, '2':0.95, '3':0.9, '4':0.85, '5':0.8, '6':0.75, '7':0.7, '0':1.0}

    d = []
    for i in range(1, 5):
        d.append(int(config[i]))

    e = [e_map[config[6]]]
    for i in range(6, 52, 2):
        e.append([e_map[config[i]], e_map[config[i+1]]])
        # e.append([1.0, 1.0])
    print(d)
    print(e)
    net.set_active_subnet(d=d, e=e)


def validate(subnet, verbose=True):
    run_manager.reset_running_statistics(net=subnet)
    _, top1, top5 = run_manager.validate(net=subnet)

    return top1


if __name__ == '__main__':
    submit_step = None # True  None
    if submit_step is None:
        if args.distributed is True:
            os.makedirs(args.path, exist_ok=True)

            # Initialize Horovod
            hvd.init()
            # Pin GPU to be used to process local rank (one GPU per process)
            torch.cuda.set_device(hvd.local_rank())

            num_gpus = hvd.size()

            torch.manual_seed(args.manual_seed)
            torch.cuda.manual_seed_all(args.manual_seed)
            np.random.seed(args.manual_seed)
            random.seed(args.manual_seed)

            # image size
            args.image_size = [int(img_size) for img_size in args.image_size.split(',')]
            if len(args.image_size) == 1:
                args.image_size = args.image_size[0]
            MyRandomResizedCrop.CONTINUOUS = args.continuous_size
            MyRandomResizedCrop.SYNC_DISTRIBUTED = not args.not_sync_distributed_image_size

            args.train_batch_size = args.base_batch_size
            args.test_batch_size = args.base_batch_size
            run_config = DistributedImageNetRunConfig(**args.__dict__, num_replicas=num_gpus, rank=hvd.rank())

            # print run config information
            if hvd.rank() == 0:
                print('Run config:')
                for k, v in run_config.config.items():
                    print('\t%s: %s' % (k, v))

            net = OFAResNet48(
                n_classes=1000, dropout_rate=0, width_mult_list=1.0, ks_list=[3],
                expand_ratio_list=[1.0], depth_list=[[5, 2], [8, 2]],
                compound=False, fixed_kernel=True)

            """ Distributed RunManager """
            # Horovod: (optional) compression algorithm.
            compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
            distributed_run_manager = DistributedRunManager(
                args.path, net, run_config, compression, backward_steps=args.dynamic_batch_size, is_root=(hvd.rank() == 0)
            )

            distributed_run_manager.load_model(args.teacher_path)
            # hvd broadcast
            distributed_run_manager.broadcast()

            # valide
            with open("CVPR_2022_NAS_Track1_test.json") as json_file:
                with open("CVPR_2022_NAS_Track1_test_submit.json", "w") as submit_json_file:
                    arch_dict = json.load(json_file)
                    # print(config)
                    # for arch in random.sample(arch_dict.keys(), 500):
                    for arch_idx, arch in enumerate(arch_dict.keys()):
                        print("-"*20, arch, "-"*20)
                        config = arch_dict[arch]
                        set_subnet(config=config['arch'])
                        distributed_run_manager.reset_running_statistics()
                        _, top1, _ = distributed_run_manager.validate()
                        arch_dict[arch]['acc'] = top1
                        print(arch_dict[arch])
                        # save to submit
                        if arch_idx % 50 == 0:
                            json.dump(arch_dict, submit_json_file)

        else:
            net = OFAResNet48(
                    n_classes=1000, dropout_rate=0, width_mult_list=1.0, ks_list=[3],
                    expand_ratio_list=[1.0], depth_list=[[5, 2], [8, 2]],
                    compound=False, fixed_kernel=True)
            net.load_weights_from_net(torch.load(args.teacher_path, map_location='cpu')['state_dict'])
            net.cuda()

            # net = nn.DataParallel(net.cuda(), device_ids=gpus, output_device=gpus[0])
            # net = nn.DataParallel(net.cuda())

            run_config = ImagenetRunConfig(
                test_batch_size=args.base_batch_size, n_worker=args.n_worker)

            run_manager = RunManager(
                '.tmp/eval_subnet', net, run_config, init=False)
            run_config.data_provider.assign_active_img_size(224)

            # with open("CVPR_2022_NAS_Track1_test.json") as json_file:
            with open(os.path.join("result", "CVPR_2022_NAS_Track1_test_submit.json")) as json_file:
                arch_dict = json.load(json_file)
                    # print(config)
                    # for arch in random.sample(arch_dict.keys(), 500):
                for arch_idx, arch in enumerate(arch_dict.keys()):
                    if arch_idx in range(args.arch_idx_start, args.arch_idx_stop):
                        config = arch_dict[arch]
                        # if config['acc'] < 20.0:
                        print("-" * 20, arch, "-" * 20)
                        set_subnet(config=config['arch'])
                        top1 = validate(net)
                        arch_dict[arch]['acc'] = top1
                        print(arch_dict[arch])
                        # save to submit
                        # if args.gpu_num == "single":
                        #     with open(os.path.join("result", "CVPR_2022_NAS_Track1_test_submit.json"),
                        #               "w") as submit_json_file:
                        #         json.dump(arch_dict, submit_json_file)
                        # elif args.gpu_num == "multiple":
                        if arch_idx % 10 == 0 or arch_idx == (args.arch_idx_stop-1):
                            with open(os.path.join("result", "CVPR_2022_NAS_Track1_test_submit_3-{}.json".format(split_idx)),
                                      "w") as submit_json_file:
                                json.dump(arch_dict, submit_json_file)
            print("-"*5, "CVPR_2022_NAS_Track1_test_submit_{}.json".format(split_idx), "Finish", "-"*5)
    elif submit_step is True:
        with open("CVPR_2022_NAS_Track1_test_submit.json", "r") as json_file:
            arch_dict = json.load(json_file)
            # print(config)
            # for arch in random.sample(arch_dict.keys(), 500):
            for split_idx in range(0, 3):
                with open(os.path.join("result", "CVPR_2022_NAS_Track1_test_submit_{}.json".format(split_idx))) as submit_json_file:
                    arch_dict_split = json.load(submit_json_file)
                arch_idx_start = split_idx * 6500
                arch_idx_stop = (split_idx + 1) * 6500
                if arch_idx_stop > 45000:
                    arch_idx_stop = 45000
                print("-" * 20, arch_idx_start, "~", arch_idx_stop, "-" * 20)
                for arch_idx, arch in enumerate(list(arch_dict_split.keys())[arch_idx_start:arch_idx_stop]):
                    arch_dict[arch]['acc'] = arch_dict_split[arch]['acc']
                    # if arch_dict[arch]['acc'] < 20:
                        # arch_dict[arch]['acc'] = (np.random.randn(1) * 5 + 70)[0]
                    # elif arch_dict[arch]['acc'] > 80:
                    #     arch_dict[arch]['acc'] = arch_dict[arch]['acc'] - 20
                    print(arch_dict[arch])
                    # save to submit
        with open("CVPR_2022_NAS_Track1_test_submit.json", "w") as json_file:
            json.dump(arch_dict, json_file)
        print("-" * 5, "CVPR_2022_NAS_Track1_test_submit.json", "Finish", "-" * 5)



        # sub_split_idx = 2
        # with open(os.path.join("result", "CVPR_2022_NAS_Track1_test_submit_{}.json".format(sub_split_idx)), "r") as json_file:
        #     arch_dict = json.load(json_file)
        #     # print(config)
        #     # for arch in random.sample(arch_dict.keys(), 500):
        #     for split_idx in range(0, 5):
        #         with open(os.path.join("result", "CVPR_2022_NAS_Track1_test_submit_{}-{}.json".format(sub_split_idx, split_idx))) as submit_json_file:
        #             arch_dict_split = json.load(submit_json_file)
        #         arch_idx_start = 1300 * split_idx + 6500 * sub_split_idx
        #         arch_idx_stop = 1300 * (split_idx + 1) + 6500 * sub_split_idx
        #         if arch_idx_stop > 45000:
        #             arch_idx_stop = 45000
        #         print("-" * 20, arch_idx_start, "~", arch_idx_stop, "-" * 20)
        #         for arch_idx, arch in enumerate(list(arch_dict_split.keys())[arch_idx_start:arch_idx_stop]):
        #             arch_dict[arch]['acc'] = arch_dict_split[arch]['acc']
        #             print(arch_dict[arch])
        #             # save to submit
        # with open(os.path.join("result", "CVPR_2022_NAS_Track1_test_submit_{}.json".format(sub_split_idx)), "w") as json_file:
        #     json.dump(arch_dict, json_file)
        # print("-" * 5, "CVPR_2022_NAS_Track1_test_submit.json", "Finish", "-" * 5)