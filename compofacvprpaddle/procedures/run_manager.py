# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import os
import time
import json
import math
from tqdm import tqdm

import numpy as np

import paddle.nn as nn
import paddle.nn.functional as F
# import paddle.DataParallel
import paddle.optimizer
import paddle.vision


# from imagenet_codebase.utils import *
from ..utils import cross_entropy_loss_with_soft_target, cross_entropy_with_label_smoothing
from ..utils.evaluation_utils import accuracy
from ..log_utils.meter import AverageMeter
from .run_config import RunConfig

class RunManager:

    def __init__(self, path, net, run_config: RunConfig, init=True, measure_latency=None, no_gpu=False, mix_prec=None):
        self.path = path
        self.net = net
        self.run_config = run_config
        self.mix_prec = mix_prec

        self.best_acc = 0
        self.start_epoch = 0

        os.makedirs(self.path, exist_ok=True)

        if init:
            self.network.init_model(run_config.model_init)

        # net info
        # net_info = get_net_info(self.net, self.run_config.data_provider.data_shape, measure_latency, True)
        with open('%s/net_info.txt' % self.path, 'w') as fout:
            # fout.write(json.dumps(net_info, indent=4) + '\n')
            try:
                fout.write(self.network.module_str)
            except Exception:
                pass

        # criterion
        if isinstance(self.run_config.mixup_alpha, float):
            self.train_criterion = cross_entropy_loss_with_soft_target
        elif self.run_config.label_smoothing > 0:
            self.train_criterion = lambda pred, target: \
                cross_entropy_with_label_smoothing(pred, target, self.run_config.label_smoothing)
        else:
            self.train_criterion = nn.CrossEntropyLoss()
        self.test_criterion = nn.CrossEntropyLoss()

        # optimizer
        if self.run_config.no_decay_keys:
            keys = self.run_config.no_decay_keys.split('#')
            net_params = [
                self.network.get_parameters(keys, mode='exclude'),  # parameters with weight decay
                self.network.get_parameters(keys, mode='include'),  # parameters without weight decay
            ]
        else:
            try:
                net_params = self.network.weight_parameters()
            except Exception:
                net_params = self.network.parameters()
        self.optimizer = self.run_config.build_optimizer(net_params)

        self.net = paddle.DataParallel(self.net)

        # self.tensorboard_logger = SummaryWriter(log_dir=os.path.join(self.path, 'tboard'))

    """ save path and log path """

    @property
    def save_path(self):
        if self.__dict__.get('_save_path', None) is None:
            save_path = os.path.join(self.path, 'checkpoint')
            os.makedirs(save_path, exist_ok=True)
            self.__dict__['_save_path'] = save_path
        return self.__dict__['_save_path']

    @property
    def logs_path(self):
        if self.__dict__.get('_logs_path', None) is None:
            logs_path = os.path.join(self.path, 'logs')
            os.makedirs(logs_path, exist_ok=True)
            self.__dict__['_logs_path'] = logs_path
        return self.__dict__['_logs_path']

    @property
    def network(self):
        return self.net

    @network.setter
    def network(self, new_val):
        self.net = new_val

    def write_log(self, log_str, prefix='valid', should_print=True):
        """ prefix: valid, train, test """
        if prefix in ['valid', 'test']:
            with open(os.path.join(self.logs_path, 'valid_console.txt'), 'a') as fout:
                fout.write(log_str + '\n')
                fout.flush()
        if prefix in ['valid', 'test', 'train']:
            with open(os.path.join(self.logs_path, 'train_console.txt'), 'a') as fout:
                if prefix in ['valid', 'test']:
                    fout.write('=' * 10)
                fout.write(log_str + '\n')
                fout.flush()
        else:
            with open(os.path.join(self.logs_path, '%s.txt' % prefix), 'a') as fout:
                fout.write(log_str + '\n')
                fout.flush()
        if should_print:
            print(log_str)

    """ save and load models """

    def save_model(self, checkpoint=None, is_best=False, model_name=None):
        if checkpoint is None:
            checkpoint = {'state_dict': self.network.state_dict()}

        if model_name is None:
            model_name = 'checkpoint.pth.tar'

        if self.mix_prec is not None:
            from apex import amp
            checkpoint['amp'] = amp.state_dict()

        checkpoint['dataset'] = self.run_config.dataset  # add `dataset` info to the checkpoint
        latest_fname = os.path.join(self.save_path, 'latest.txt')
        model_path = os.path.join(self.save_path, model_name)
        with open(latest_fname, 'w') as fout:
            fout.write(model_path + '\n')
        paddle.save(checkpoint, model_path)

        if is_best:
            best_path = os.path.join(self.save_path, 'model_best.pth.tar')
            paddle.save({'state_dict': checkpoint['state_dict']}, best_path)

    def load_model(self, model_fname=None):
        latest_fname = os.path.join(self.save_path, 'latest.txt')
        if model_fname is None and os.path.exists(latest_fname):
            with open(latest_fname, 'r') as fin:
                model_fname = fin.readline()
                if model_fname[-1] == '\n':
                    model_fname = model_fname[:-1]
        try:
            if model_fname is None or not os.path.exists(model_fname):
                model_fname = '%s/checkpoint.pth.tar' % self.save_path
                with open(latest_fname, 'w') as fout:
                    fout.write(model_fname + '\n')
            print("=> loading checkpoint '{}'".format(model_fname))

            if paddle.device.cuda.device_count() > 0:
                checkpoint = paddle.load(model_fname)
            else:
                checkpoint = paddle.load(model_fname, map_location='cpu')

            self.network.load_state_dict(checkpoint['state_dict'])

            if 'epoch' in checkpoint:
                self.start_epoch = checkpoint['epoch'] + 1
            if 'best_acc' in checkpoint:
                self.best_acc = checkpoint['best_acc']
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.mix_prec is not None and 'amp' in checkpoint:
                from apex import amp
                amp.load_state_dict(checkpoint['amp'])

            print("=> loaded checkpoint '{}'".format(model_fname))
        except Exception:
            print('fail to load checkpoint from %s' % self.save_path)

    def save_config(self):
        """ dump run_config and net_config to the model_folder """
        net_save_path = os.path.join(self.path, 'net.config')
        json.dump(self.network.config, open(net_save_path, 'w'), indent=4)
        print('Network configs dump to %s' % net_save_path)

        run_save_path = os.path.join(self.path, 'run.config')
        json.dump(self.run_config.config, open(run_save_path, 'w'), indent=4)
        print('Run configs dump to %s' % run_save_path)

    """ train and test """

    def validate(self, epoch=0, is_test=True, run_str='', net=None, data_loader=None, no_logs=False):
        if net is None:
            net = self.net
        net = paddle.DataParallel(net)

        if data_loader is None:
            if is_test:
                data_loader = self.run_config.test_loader
            else:
                data_loader = self.run_config.valid_loader

        net.eval()

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        with tqdm(total=len(data_loader),
                  desc='Validate Epoch #{} {}'.format(epoch + 1, run_str), disable=no_logs) as t:
            for i, (images, labels) in enumerate(data_loader):
                if i >10:
                    break

                images, labels = images.to(self.device), labels.to(self.device)
                # compute output
                output = net(images)
                loss = self.test_criterion(output, labels)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))

                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0].item(), images.size(0))
                top5.update(acc5[0].item(), images.size(0))
                t.set_postfix({
                    'loss': losses.avg,
                    'top1': top1.avg,
                    'top5': top5.avg,
                    'img_size': images.size(2),
                })
                t.update(1)
        return losses.avg, top1.avg, top5.avg

    def validate_all_resolution(self, epoch=0, is_test=True, net=None):
        if net is None:
            net = self.network
        if isinstance(self.run_config.data_provider.image_size, list):
            img_size_list, loss_list, top1_list, top5_list = [], [], [], []
            for img_size in self.run_config.data_provider.image_size:
                img_size_list.append(img_size)
                self.run_config.data_provider.assign_active_img_size(img_size)
                self.reset_running_statistics(net=net)
                loss, top1, top5 = self.validate(epoch, is_test, net=net)
                loss_list.append(loss)
                top1_list.append(top1)
                top5_list.append(top5)
            return img_size_list, loss_list, top1_list, top5_list
        else:
            loss, top1, top5 = self.validate(epoch, is_test, net=net)
            return [self.run_config.data_provider.active_img_size], [loss], [top1], [top5]

    def train_one_epoch(self, args, epoch, warmup_epochs=0, warmup_lr=0):
        # switch to train mode
        self.net.train()

        nBatch = len(self.run_config.train_loader)

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        data_time = AverageMeter()

        with tqdm(total=nBatch,
                  desc='Train Epoch #{}'.format(epoch + 1)) as t:
            end = time.time()
            for i, (images, labels) in enumerate(self.run_config.train_loader):
                data_time.update(time.time() - end)
                if epoch < warmup_epochs:
                    new_lr = self.run_config.warmup_adjust_learning_rate(
                        self.optimizer, warmup_epochs * nBatch, nBatch, epoch, i, warmup_lr,
                    )
                else:
                    new_lr = self.run_config.adjust_learning_rate(self.optimizer, epoch - warmup_epochs, i, nBatch)

                images, labels = images.to(self.device), labels.to(self.device)
                target = labels

                # soft target
                if args.teacher_model is not None:
                    args.teacher_model.train()
                    soft_logits = args.teacher_model(images).detach()
                    soft_label = F.softmax(soft_logits, dim=1)

                # compute output
                output = self.net(images)
                loss = self.train_criterion(output, labels)

                if args.teacher_model is None:
                    loss_type = 'ce'
                else:
                    if args.kd_type == 'ce':
                        kd_loss = cross_entropy_loss_with_soft_target(output, soft_label)
                    else:
                        kd_loss = F.mse_loss(output, soft_logits)
                    loss = args.kd_ratio * kd_loss + loss
                    loss_type = '%.1fkd-%s & ce' % (args.kd_ratio, args.kd_type)

                # compute gradient and do SGD step
                self.net.zero_grad()  # or self.optimizer.zero_grad()
                if self.mix_prec is not None:
                    from apex import amp
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                self.optimizer.step()

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0].item(), images.size(0))
                top5.update(acc5[0].item(), images.size(0))

                t.set_postfix({
                    'loss': losses.avg,
                    'top1': top1.avg,
                    'top5': top5.avg,
                    'img_size': images.size(2),
                    'lr': new_lr,
                    'loss_type': loss_type,
                    'data_time': data_time.avg,
                })
                t.update(1)
                end = time.time()
        return losses.avg, top1.avg, top5.avg

    def train(self, args, warmup_epoch=0, warmup_lr=0):
        for epoch in range(self.start_epoch, self.run_config.n_epochs + warmup_epoch):
            train_loss, train_top1, train_top5 = self.train_one_epoch(args, epoch, warmup_epoch, warmup_lr)

            if (epoch + 1) % self.run_config.validation_frequency == 0:
                img_size, val_loss, val_acc, val_acc5 = self.validate_all_resolution(epoch=epoch, is_test=False)

                is_best = np.mean(val_acc) > self.best_acc
                self.best_acc = max(self.best_acc, np.mean(val_acc))
                val_log = 'Valid [{0}/{1}]\tloss {2:.3f}\ttop-1 acc {3:.3f} ({4:.3f})'. \
                    format(epoch + 1 - warmup_epoch, self.run_config.n_epochs,
                           np.mean(val_loss), np.mean(val_acc), self.best_acc)
                val_log += '\ttop-5 acc {0:.3f}\tTrain top-1 {top1:.3f}\tloss {train_loss:.3f}\t'. \
                    format(np.mean(val_acc5), top1=train_top1, train_loss=train_loss)
                for i_s, v_a in zip(img_size, val_acc):
                    val_log += '(%d, %.3f), ' % (i_s, v_a)
                self.write_log(val_log, prefix='valid', should_print=False)
            else:
                is_best = False

            self.save_model({
                'epoch': epoch,
                'best_acc': self.best_acc,
                'optimizer': self.optimizer.state_dict(),
                'state_dict': self.network.state_dict(),
            }, is_best=is_best)

    def reset_running_statistics(self, net=None):
        from ..models.dynamic_nn_utils import set_running_statistics
        if net is None:
            net = self.network
        sub_train_loader = self.run_config.random_sub_train_loader(2000, 100)
        set_running_statistics(net, sub_train_loader)

    # def log_to_tensorboard(self, metric_key, metric_value, step):
    #     self.tensorboard_logger.add_scalar(metric_key, metric_value, step)
