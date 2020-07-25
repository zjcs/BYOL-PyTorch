#-*- coding:utf-8 -*-
import os
import torch
from torchvision import transforms, datasets
from .byol_transform import MultiViewDataInjector, GaussianBlur, Solarize


class ImageNetLoader():
    def __init__(self, config):
        self.image_dir = config['data']['image_dir']
        self.num_replicas = config['world_size']
        self.rank = config['rank']
        self.distributed = config['distributed']
        self.resize_size = config['data']['resize_size']
        self.data_workers = config['data']['data_workers']
        self.dual_views = config['data']['dual_views']
        self.s = 1

    def get_transform(self, stage, gb_prob=1.0, solarize_prob=0.):
        t_list = []
        color_jitter = transforms.ColorJitter(0.4 * self.s, 0.4 * self.s, 0.2 * self.s, 0.1 * self.s)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if stage in ('train', 'val'):
            t_list = [transforms.RandomResizedCrop(size=self.resize_size),
                      transforms.RandomHorizontalFlip(),
                      transforms.RandomApply([color_jitter], p=0.8),
                      transforms.RandomGrayscale(p=0.2),
                      transforms.RandomApply([GaussianBlur(kernel_size=23)], p=gb_prob),
                      transforms.RandomApply([Solarize()], p=solarize_prob),
                      transforms.ToTensor(),
                      normalize]
        elif stage == 'ft':
            t_list = [transforms.RandomResizedCrop(size=self.resize_size),
                      transforms.RandomHorizontalFlip(),
                      transforms.ToTensor(),
                      normalize]
        elif stage == 'test':
            t_list = [transforms.Resize(256),
                      transforms.CenterCrop(224),
                      transforms.ToTensor(),
                      normalize]
        transform = transforms.Compose(t_list)
        return transform

    def get_loader(self, stage, batch_size):
        dataset = self.get_dataset(stage)
        if self.distributed and stage in ('train', 'ft'):
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=self.num_replicas, rank=self.rank)
        else:
            self.train_sampler = None

        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=(self.train_sampler is None and stage not in ('val', 'test')),
            num_workers=self.data_workers,
            pin_memory=True,
            sampler=self.train_sampler,
            drop_last=True
        )
        return data_loader

    def get_dataset(self, stage):
        image_dir = self.image_dir + f"imagenet_{'train' if stage in ('train', 'ft') else 'val'}"
        transform1 = self.get_transform(stage)
        if self.dual_views:
            transform2 = self.get_transform(stage, gb_prob=0.1, solarize_prob=0.2)
            transform = MultiViewDataInjector([transform1, transform2])
        else:
            transform = transform1
        dataset = datasets.ImageFolder(image_dir, transform=transform)
        return dataset

    def set_epoch(self, epoch):
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
