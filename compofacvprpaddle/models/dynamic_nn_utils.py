# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import copy

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ..log_utils.meter import AverageMeter
from ..utils import DistributedTensor
from .dynamic_op import DynamicBatchNorm2d


def set_running_statistics(model, data_loader, distributed=False):
    bn_mean = {}
    bn_var = {}
    
    forward_model = copy.deepcopy(model)
    for name, m in forward_model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            if distributed:
                bn_mean[name] = DistributedTensor(name + '#mean')
                bn_var[name] = DistributedTensor(name + '#var')
            else:
                bn_mean[name] = AverageMeter()
                bn_var[name] = AverageMeter()

            def new_forward(bn, mean_est, var_est):
                def lambda_forward(x):
                    batch_mean = x.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)  # 1, C, 1, 1
                    batch_var = (x - batch_mean) * (x - batch_mean)
                    batch_var = batch_var.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
                    
                    batch_mean = paddle.squeeze(batch_mean)
                    batch_var = paddle.squeeze(batch_var)

                    mean_est.update(batch_mean.data, x.size(0))
                    var_est.update(batch_var.data, x.size(0))

                    # bn forward using calculated mean & var
                    _feature_dim = batch_mean.size(0)

                    return F.batch_norm(
                        x, batch_mean, batch_var, bn.weight[:_feature_dim],
                        bn.bias[:_feature_dim], False,
                        0.0, bn.eps,
                    )

                return lambda_forward

            m.forward = new_forward(m, bn_mean[name], bn_var[name])

    DynamicBatchNorm2d.SET_RUNNING_STATISTICS = True
    for images, labels in data_loader:
        # images = images.to(get_net_device(forward_model))
        forward_model(images)
    DynamicBatchNorm2d.SET_RUNNING_STATISTICS = False
    
    for name, m in model.named_modules():
        if name in bn_mean and bn_mean[name].count > 0:
            feature_dim = bn_mean[name].avg.size(0)
            assert isinstance(m, nn.BatchNorm2d)
            m.running_mean.data[:feature_dim].copy_(bn_mean[name].avg)
            m.running_var.data[:feature_dim].copy_(bn_var[name].avg)


def adjust_bn_according_to_idx(bn, idx):
    bn.weight.data = paddle.index_select(bn.weight.data, 0, idx)
    bn.bias.data = paddle.index_select(bn.bias.data, 0, idx)
    bn.running_mean.data = paddle.index_select(bn.running_mean.data, 0, idx)
    bn.running_var.data = paddle.index_select(bn.running_var.data, 0, idx)


def copy_bn(target_bn, src_bn):
    feature_dim = target_bn.num_features
    
    target_bn.weight.data.copy_(src_bn.weight.data[:feature_dim])
    target_bn.bias.data.copy_(src_bn.bias.data[:feature_dim])
    target_bn.running_mean.data.copy_(src_bn.running_mean.data[:feature_dim])
    target_bn.running_var.data.copy_(src_bn.running_var.data[:feature_dim])
