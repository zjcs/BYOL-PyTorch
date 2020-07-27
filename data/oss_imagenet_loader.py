#-*- coding:utf-8 -*-
import os
import math
import torch
import torchvision.transforms as transforms
import oss2 as oss
from PIL import Image
import warnings
from .byol_transform import MultiViewDataInjector, GaussianBlur, Solarize
warnings.simplefilter("ignore", UserWarning)
oss.logger.setLevel(30)


class OssImageDataset(torch.utils.data.Dataset):
    def __init__(self, stage, transform, endpoint, key_id, secret_id, bucket, image_root, list_file, resize_size, rank, epoch, num_replicas):
        self.transform = transform
        # oss init
        self.endpoint = endpoint
        self.key_id = key_id
        self.secret_id = secret_id
        self.bucket = bucket
        self.init_oss()
        # image about
        self.image_root = image_root
        self.list_file = list_file
        self.resize_size = resize_size
        # data scale about
        self.rank = rank
        self.num_replicas = num_replicas
        self.epoch = epoch
        self.stage = stage
        if not self.stage == 'prepare':
            self.collect_data()

    def init_oss(self):
        auth = oss.Auth(self.key_id, self.secret_id)
        try:
            self.bucket = oss.Bucket(auth, self.endpoint, self.bucket, connect_timeout=300)
        except:
            raise ValueError("Oss server is unreachable!")

    def default_transform(self):
        t_list = [transforms.ToTensor()]
        transform = transforms.Compose(t_list)
        return transform

    def collect_data(self):
        lines = []
        for _ in range(5):
            try:
                for l in str(self.bucket.get_object(self.list_file).read(), 'utf-8').split('\n'):
                    if l: lines.append(l)
                break
            except:
                continue

        # print(len(lines))

        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(len(lines), generator=g).tolist()

        num_samples = int(math.ceil(len(lines) * 1.0 / self.num_replicas))
        total_size = num_samples * self.num_replicas

        indices += indices[:(total_size - len(indices))]
        assert len(indices) == total_size

        # subsample
        indices = set(indices[self.rank:total_size:self.num_replicas])
        assert len(indices) == num_samples

        print(f'Epoch: {self.epoch}, rank: {self.rank}, samples number: {len(indices)}')

        pairs = []
        for i, line in enumerate(lines):
            if i not in indices: continue
            file_path, label = line.strip().split(' ')
            oss_key = os.path.join(self.image_root, file_path)
            pairs.append((oss_key, int(label)))
        # print(len(pairs))
        # exit()
        self.pairs = pairs

    def __getitem__(self, index):
        oss_key, label = self.pairs[index]

        for _ in range(10):
            try:
                image = Image.open(self.bucket.get_object(oss_key)).convert('RGB')
                break
            except:
                print(f'oss: {oss_key} failed, tried again')

        if self.transform is not None:
            image = self.transform(image)
        else:
            transform = self.default_transform()
            image = transform(image)
        return image, label

    def __len__(self):
        return len(self.pairs)

class OssImageLoader():
    def __init__(self, config):
        self.endpoint = config['data']['endpoint']
        self.key_id = config['data']['key_id']
        self.secret_id = config['data']['secret_id']
        self.bucket = config['data']['bucket']
        self.train_root = config['data']['train_root']
        self.val_root = config['data']['val_root']
        self.train_list = config['data']['train_list']
        self.val_list = config['data']['val_list']
        self.num_replicas = config['world_size']
        self.rank = config['rank']
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

    def get_loader(self, stage, batch_size, epoch=0):
        dataset = self.get_dataset(stage, epoch=epoch)
        print(f'{stage} dataset size: {len(dataset)}')

        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=(stage not in ('val', 'test')),
            num_workers=self.data_workers,
            pin_memory=True,
            drop_last=True
        )
        return data_loader

    def get_dataset(self, stage, epoch=0):
        if stage in ('train', 'ft'):
            image_root = self.train_root
            list_file = self.train_list
        else:
            image_root = self.val_root
            list_file = self.val_list
        transform1 = self.get_transform(stage)
        if self.dual_views:
            transform2 = self.get_transform(stage, gb_prob=0.1, solarize_prob=0.2)
            transform = MultiViewDataInjector([transform1, transform2])
        else:
            transform = transform1
        dataset = OssImageDataset(stage, transform,
                                  self.endpoint, self.key_id, self.secret_id, self.bucket,
                                  image_root, list_file, self.resize_size,
                                  self.rank, epoch, self.num_replicas)
        return dataset
