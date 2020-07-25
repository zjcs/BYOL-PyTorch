#-*- coding:utf-8 -*-
from io import BytesIO
import time
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import oss2 as oss

import apex
from apex.parallel import DistributedDataParallel as DDP
from apex import amp

from model import ResNetwithProjection, Predictor
from optimizer.LARSSGD import LARS
from data.imagenet_loader import ImageNetLoader
from data.oss_imagenet_loader import OssImageLoader
from utils.logging_util import log_tool
from utils import params_util, logging_util
import utils.evaluation_util as eval_util
from utils.data_prefetcher import data_prefetcher


class BYOLTrainer():
    def __init__(self, config):
        self.config = config
        self.time_stamp = self.config['checkpoint'].get('time_stamp',
            datetime.datetime.now().strftime('%m%d_%H-%M'))

        """device parameters"""
        self.world_size = self.config['world_size']
        self.rank = self.config['rank']
        self.gpu = self.config['local_rank']
        self.distributed = self.config['distributed']

        """get the train parameters!"""
        self.total_epochs = self.config['optimizer']['total_epochs']
        self.warmup_epochs = self.config['optimizer']['warmup_epochs']

        self.train_batch_size = self.config['data']['train_batch_size']
        self.val_batch_size = self.config['data']['val_batch_size']
        self.global_batch_size = self.world_size * self.train_batch_size

        self.warmup_steps = self.warmup_epochs * 1281167 // self.global_batch_size
        self.total_steps = self.total_epochs * 1281167 // self.global_batch_size

        base_lr = self.config['optimizer']['base_lr'] / 256
        self.max_lr = base_lr * self.global_batch_size

        self.base_mm = self.config['model']['base_momentum']

        """construct the whole network"""
        self.resume_path = self.config['checkpoint']['resume_path']
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.gpu}')
            torch.cuda.set_device(self.device)
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')
        self.construct_model()

        """save oss path"""
        self.save_epoch = self.config['checkpoint']['save_epoch']
        self.ckpt_prefix = self.config['checkpoint']['ckpt_prefix'].format(
            self.time_stamp, self.config['model']['backbone']['type'], {})
        ckpt_endpoint = self.config['checkpoint']['ckpt_endpoint']
        ckpt_key_id = self.config['checkpoint']['ckpt_key_id']
        ckpt_secret_id = self.config['checkpoint']['ckpt_secret_id']
        ckpt_bucket = self.config['checkpoint']['ckpt_bucket']

        auth = oss.Auth(ckpt_key_id, ckpt_secret_id)
        try:
            self.ckpt_bucket = oss.Bucket(auth, ckpt_endpoint, ckpt_bucket, connect_timeout=300)
        except:
            raise ValueError("Oss server is unreachable!")

        """log tools in the running phase"""
        self.log_step = self.config['log']['log_step']
        self.logger = eval_util.LogCollector()
        self.logging = logging_util.get_std_logging()
        self.steps = 0

        if self.rank == 0:
            self.setup_oss_log_files()

    def setup_oss_log_files(self):

        self.loss_log_file = self.config['log']['loss_log_file'].format(self.time_stamp)
        self.loss_log_png = self.config['log']['loss_log_png'].format(self.time_stamp)
        self.loss_log_tool = log_tool(bucket=self.ckpt_bucket, log_path=self.loss_log_file)

        self.lr_log_file = self.config['log']['lr_log_file'].format(self.time_stamp)
        self.lr_log_png = self.config['log']['lr_log_png'].format(self.time_stamp)
        self.lr_log_tool = log_tool(bucket=self.ckpt_bucket, log_path=self.lr_log_file)

    def wrap_model(self, net, sync_bn=False):
        if sync_bn:
            net = apex.parallel.convert_syncbn_model(net)
        net = net.to(self.device)
        return net

    def construct_model(self):
        # get data instance
        self.stage = self.config['stage']
        assert self.stage == 'train', ValueError(f'Invalid stage: {self.stage}, only "train" for BYOL training')
        self.use_local_dataloader = self.config['data']['use_local_dataloader']
        if self.use_local_dataloader:
            self.data_ins = ImageNetLoader(self.config)
            self.train_loader = self.data_ins.get_loader(self.stage, self.train_batch_size)
        else:
            self.data_ins = OssImageLoader(self.config)

        self.sync_bn = self.config['amp']['sync_bn']
        self.opt_level = self.config['amp']['opt_level']
        print(f"sync_bn: {self.sync_bn}")

        print("init online network!")
        online_network = ResNetwithProjection(self.config)
        self.online_network = self.wrap_model(online_network, sync_bn=self.sync_bn)
        print("init online network end!")

        print("init predictor!")
        predictor = Predictor(self.config)
        self.predictor = self.wrap_model(predictor, sync_bn=self.sync_bn)
        print("init predictor end!")

        print("init target network!")
        target_network = ResNetwithProjection(self.config)
        self.target_network = self.wrap_model(target_network, sync_bn=self.sync_bn)
        print("init target network end!")

        self.initializes_target_network()

        # optimizer
        print("get optimizer!")
        momentum = self.config['optimizer']['momentum']
        weight_decay = self.config['optimizer']['weight_decay']
        exclude_bias_and_bn = self.config['optimizer']['exclude_bias_and_bn']
        params = params_util.collect_params([self.online_network, self.predictor],
                                            exclude_bias_and_bn=exclude_bias_and_bn)
        self.optimizer = LARS(params, lr=self.max_lr, momentum=momentum, weight_decay=weight_decay)

        # amp
        print("amp init!")
        (self.online_network, self.predictor), self.optimizer = amp.initialize(
            [self.online_network, self.predictor], self.optimizer, opt_level=self.opt_level)

        if self.distributed:
            self.online_network = DDP(self.online_network, delay_allreduce=True)
            self.predictor = DDP(self.predictor, delay_allreduce=True)
            self.target_network = DDP(self.target_network, delay_allreduce=True)
        print("init net end!")

    @torch.no_grad()
    def initializes_target_network(self):
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False     # not update by gradient

    # resume snapshots from pre-train
    def resume_model(self, model_path=None):
        if model_path is None and not self.resume_path:
            self.start_epoch = 0
            self.logging.info("--> No loaded checkpoint!")
        else:
            model_path = model_path or self.resume_path
            model_data = self.ckpt_bucket.get_object(model_path).read()
            checkpoint = torch.load(BytesIO(model_data), map_location=self.device)

            self.start_epoch = checkpoint['epoch']
            self.online_network.load_state_dict(checkpoint['online_network'], strict=True)
            self.predictor.load_state_dict(checkpoint['predictor'], strict=True)
            self.target_network.load_state_dict(checkpoint['target_network'], strict=True)
            amp.load_state_dict(checkpoint['amp'])
            self.steps = checkpoint['steps']
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.logging.info(
                "--> loaded checkpoint '{}' (epoch {})".format(model_path, self.start_epoch))

    # save snapshots
    def save_checkpoint(self, epoch):
        if epoch % self.save_epoch == 0 and self.rank == 0:
            state = {'config': self.config,
                     'epoch': epoch,
                     'online_network': self.online_network.state_dict(),
                     'predictor': self.predictor.state_dict(),
                     'target_network': self.target_network.state_dict(),
                     'steps': self.steps,
                     'optimizer': self.optimizer.state_dict(),
                     'amp': amp.state_dict()
                     }
            prefix = self.ckpt_prefix.format(epoch)
            snapshot_buf = BytesIO()
            torch.save(state, snapshot_buf)
            for _ in range(5):
                try:
                    self.ckpt_bucket.put_object(prefix, snapshot_buf.getvalue())
                    break
                except:
                    continue

    def adjust_learning_rate(self, step):
        """learning rate warm up and decay"""
        max_lr = self.max_lr
        min_lr = 1e-3 * self.max_lr
        if step < self.warmup_steps:
            lr = (max_lr - min_lr) * step / self.warmup_steps + min_lr
        else:
            lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos((step - self.warmup_steps) * np.pi / self.total_steps))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def adjust_mm(self, step):
        self.mm = 1 - (1 - self.base_mm) * (np.cos(np.pi * step / self.total_steps) + 1) / 2

    def switch_train(self):
        # switch to train mode
        self.online_network.train()
        self.predictor.train()
        self.target_network.train()

    def switch_eval(self):
        # switch to eval mode
        self.online_network.eval()
        self.predictor.eval()
        self.target_network.eval()

    @torch.no_grad()
    def update_target_network(self):
        """Momentum update of target network"""
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.mul_(self.mm).add_(1. - self.mm, param_q.data)

    def forward_loss(self, preds, targets):
        bz = preds.size(0)
        preds_norm = F.normalize(preds, dim=1)
        targets_norm = F.normalize(targets, dim=1)
        loss = 2 - 2 * (preds_norm * targets_norm).sum() / bz
        return loss

    ## train the network in all data for one epoch
    def train_epoch(self, epoch, printer=print):
        batch_time = eval_util.AverageMeter()
        data_time = eval_util.AverageMeter()
        online_time = eval_util.AverageMeter()
        target_time = eval_util.AverageMeter()
        backward_time = eval_util.AverageMeter()
        log_time = eval_util.AverageMeter()
        end = time.time()

        if self.use_local_dataloader:
            self.data_ins.set_epoch(epoch)
        else:
            self.train_loader = self.data_ins.get_loader(self.stage,
                batch_size=self.train_batch_size, epoch=epoch)

        prefetcher = data_prefetcher(self.train_loader)
        images, _ = prefetcher.next()
        i = 0
        while images is not None:
            i += 1
            self.adjust_learning_rate(self.steps)
            self.adjust_mm(self.steps)
            self.steps += 1
            assert images.dim() == 5, f"Input must have 5 dims, got: {images.dim()}"
            view1 = images[:, 0, ...].contiguous()
            view2 = images[:, 1, ...].contiguous()
            # measure data loading time
            data_time.update(time.time() - end)

            # online forward
            tflag = time.time()
            q = self.predictor(self.online_network(torch.cat([view1, view2], dim=0)))
            online_time.update(time.time() - tflag)

            # target forward
            tflag = time.time()
            with torch.no_grad():
                target_z = self.target_network(torch.cat([view2, view1], dim=0)).detach().clone()
            target_time.update(time.time() - tflag)

            tflag = time.time()
            loss = self.forward_loss(q, target_z)

            self.optimizer.zero_grad()
            if self.opt_level == 'O0':
                loss.backward()
            else:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            self.optimizer.step()

            self.update_target_network()
            backward_time.update(time.time() - tflag)

            tflag = time.time()
            if self.steps % self.log_step == 0:
                self.logger.update('steps', self.steps)
                self.logger.update('lr', round(self.optimizer.param_groups[0]['lr'], 5))
                self.logger.update('loss', loss.item(), view1.size(0))

                if self.rank == 0:
                    self.loss_log_tool.update(self.logger.get_key_val('steps'), self.logger.get_key_val('loss'))
                    self.lr_log_tool.update(self.logger.get_key_val('steps'), self.logger.get_key_val('lr'))
                    if self.steps % 100 == 0:
                        self.loss_log_tool.plot(self.loss_log_png, x_label='steps', y_label='loss', label='loss')
                        self.loss_log_tool.save_log()
                        self.lr_log_tool.plot(self.lr_log_png, x_label='steps', y_label='lr', label='lr')
                        self.lr_log_tool.save_log()
            log_time.update(time.time() - tflag)

            batch_time.update(time.time() - end)
            end = time.time()

            # Print log info
            if self.gpu == 0 and self.steps % self.log_step == 0:
                printer(f'Epoch: [{epoch}][{i}/{len(self.train_loader)}]\t{str(self.logger)}\t'
                        f'Batch Time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                        f'Data Time {data_time.val:.4f} ({data_time.avg:.4f})\t'
                        f'online Time {online_time.val:.4f} ({online_time.avg:.4f})\t'
                        f'target Time {target_time.val:.4f} ({target_time.avg:.4f})\t'
                        f'backward Time {backward_time.val:.4f} ({backward_time.avg:.4f})\t'
                        f'Log Time {log_time.val:.4f} ({log_time.avg:.4f})\t')

            images, _ = prefetcher.next()
