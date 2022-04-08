import argparse
import numpy as np
import os
import random

import horovod.torch as hvd
import torch

from compofacvpr.models.dynamic_op import DynamicSeparableConv2d
from compofacvpr.models.cells_search.ofa_mbv3 import OFAMobileNetV3
from compofacvpr.models.cells_search.ofa_resnet48 import OFAResNet48
from compofacvpr.procedures import DistributedImageNetRunConfig
from compofacvpr.procedures.distributed_run_manager import DistributedRunManager
from compofacvpr.datasets.base_provider import MyRandomResizedCrop
from compofacvpr.utils import download_url
from compofacvpr.procedures.progressive_shrinking import load_models

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='teacher', choices=[
    'kernel', 'depth', 'expand', 'compound', 'teacher',
])
parser.add_argument('--phase', type=int, default=1, choices=[1, 2])
parser.add_argument('--name', type=str, default='default')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--fixed_kernel', action='store_false')
parser.add_argument('--heuristic', type=str, default='none', choices=['simple', 'none'])

args = parser.parse_args()
args.teacher_path = download_url('https://file.lzhu.me/projects/OnceForAll/ofa_checkpoints/ofa_D4_E6_K7',
                                 model_dir='./downloads')

args.manual_seed = 0

args.lr_schedule_type = 'cosine'

args.base_batch_size = 64
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

args.n_worker = 8
args.resize_scale = 0.08
args.distort_color = 'tf'
args.image_size = '128,160,192,224'
# args.image_size = '224'
args.continuous_size = True
args.not_sync_distributed_image_size = False

args.bn_momentum = 0.1
args.bn_eps = 1e-5
args.dropout = 0.1
args.base_stage_width = 'proxyless'

args.width_mult_list = '1.0'
args.dy_conv_scaling_mode = -1
args.independent_distributed_sampling = False

# args.kd_ratio = 1.0
args.kd_ratio = 0
args.kd_type = 'ce'


if args.task == 'kernel':
    args.path = f'runs/{args.name}/teacher2kernel'
    args.dynamic_batch_size = 1
    args.n_epochs = 120
    args.base_lr = 2.6 * args.base_batch_size/2048
    args.warmup_epochs = 5
    args.warmup_lr = -1
    args.ks_list = '3,5,7'
    args.expand_list = '6'
    args.depth_list = '4'
elif args.task == 'depth':
    args.path = f'runs/{args.name}/base2depth/phase{args.phase}'
    args.dynamic_batch_size = 2
    if args.phase == 1:
        args.n_epochs = 25
        args.base_lr = 0.08 * args.base_batch_size/2048
        args.warmup_epochs = 0
        args.warmup_lr = -1
        args.expand_list = '6'
        args.depth_list = '3,4'
        args.ks_list = '3,5,7'
    else:
        args.n_epochs = 120
        args.base_lr = 0.24*args.base_batch_size/2048
        args.warmup_epochs = 5
        args.warmup_lr = -1
        args.expand_list = '6'
        args.depth_list = '2,3,4'
        args.ks_list = '3,5,7'
elif args.task == 'expand':
    args.path = f'runs/{args.name}/depth2depth_width/phase{args.phase}'
    args.dynamic_batch_size = 4
    if args.phase == 1:
        args.n_epochs = 25
        args.base_lr = 0.08 * args.base_batch_size/2048
        args.warmup_epochs = 0
        args.warmup_lr = -1
        args.expand_list = '4,6'
        args.depth_list = '2,3,4'
        args.ks_list = '3,5,7'
    else:
        args.n_epochs = 120
        args.base_lr = 0.24*args.base_batch_size/2048
        args.warmup_epochs = 5
        args.warmup_lr = -1
        args.expand_list = '3,4,6'
        args.depth_list = '2,3,4'
        args.ks_list = '3,5,7'
elif args.task == 'teacher':
    args.path = f'runs/teacher/{args.name}'
    args.dynamic_batch_size = 1
    args.n_epochs = 180
    args.base_lr = 2.6 * args.base_batch_size / 2048
    args.warmup_epochs = 0
    args.warmup_lr = -1
    args.kd_ratio = 0.0
    args.expand_list = '1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7'  # 1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7  '1,2,3,4,5,6,7'
    args.depth_list = '5,2,8,2'  # stage 1,2,4 : 5~2, stage 3: 8~2
    args.ks_list = '3'
elif args.task == 'compound':
    assert(args.heuristic=='simple')
    # args.path = f'runs/{args.name}/compound/phase{args.phase}'
    args.path = f'runs1/{args.name}/compound/phase{args.phase}'
    args.dynamic_batch_size = 4
    if args.phase == 1:
        args.n_epochs = 25
        args.base_lr = 0.08*args.base_batch_size/2048
        args.warmup_epochs = 0
        args.warmup_lr = -1
        args.expand_list = '3,4,6'
        args.depth_list = '2,3,4'
        args.ks_list = '3,5,7'
    else:
        args.n_epochs = 120
        args.base_lr = 0.24*args.base_batch_size/2048
        args.warmup_epochs = 5
        args.warmup_lr = -1
        args.expand_list = '3,4,6'
        args.depth_list = '2,3,4'
        args.ks_list = '3,5,7'
else:
    raise NotImplementedError


if __name__ == '__main__':
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

    # build run config from args
    args.lr_schedule_param = None
    args.opt_param = {
        'momentum': args.momentum,
        'nesterov': not args.no_nesterov,
    }
    args.init_lr = args.base_lr * num_gpus  # linearly rescale the learning rate
    if args.warmup_lr < 0:
        args.warmup_lr = args.base_lr
    args.train_batch_size = args.base_batch_size
    args.test_batch_size = args.base_batch_size * 2
    run_config = DistributedImageNetRunConfig(**args.__dict__, num_replicas=num_gpus, rank=hvd.rank())

    # print run config information
    if hvd.rank() == 0:
        print('Run config:')
        for k, v in run_config.config.items():
            print('\t%s: %s' % (k, v))

    if args.dy_conv_scaling_mode == -1:
        args.dy_conv_scaling_mode = None
    DynamicSeparableConv2d.KERNEL_TRANSFORM_MODE = args.dy_conv_scaling_mode

    # build net from args
    args.width_mult_list = [float(width_mult) for width_mult in args.width_mult_list.split(',')]
    args.ks_list = [int(ks) for ks in args.ks_list.split(',')]
    args.expand_list = [float(e) for e in args.expand_list.split(',')]
    args.depth_list = [int(d) for d in args.depth_list.split(',')]
    args.depth_list = [[args.depth_list[0], args.depth_list[1]], [args.depth_list[2], args.depth_list[3]]]

    net = OFAResNet48(
        n_classes=run_config.data_provider.n_classes, bn_param=(args.bn_momentum, args.bn_eps),
        dropout_rate=args.dropout, base_stage_width=args.base_stage_width, width_mult_list=args.width_mult_list,
        ks_list=args.ks_list, expand_ratio_list=args.expand_list, depth_list=args.depth_list,
        compound=(args.heuristic!='none'), fixed_kernel=args.fixed_kernel,
    )

    # # test_in = torch.randn((64, 3, 224, 224))
    # test_in = torch.randn((64, 3, 128, 128))
    #
    # test_out = net(test_in)
    #
    # print(test_out.size())

    test_mode = "summary"
    if test_mode == "summary":
        from torchsummary import summary

        net.set_constraint([[5, 2], [8, 2]], 'depth')
        net.set_constraint([1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7], 'expand_ratio')
        net.set_constraint([3], 'kernel_size')
        net.set_active_subnet(d=[3, 4, 6, 5], e=1, ks=None)
        net.sample_active_subnet()

        net.to("cuda")

        # net.cuda()
        summary(net, (3, 208, 208))
    elif test_mode == "sample_sub_net":
        net.clear_constraint()
        print("net: ")
        print(net.sample_active_subnet())
        print(net.name())
        print(net.config)
        print(net.module_str)
