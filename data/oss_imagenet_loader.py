#-*- coding:utf-8 -*-
import math
import torch
import oss2 as oss
from PIL import Image
from .byol_transform import MultiViewDataInjector, get_transform

class OssImageDataset(torch.utils.data.Dataset):
    def __init__(self, transform, endpoint, key_id, secret_id, bucket, list_file, rank, num_replicas):
        self.transform = transform
        # oss init
        self.endpoint = endpoint
        self.key_id = key_id
        self.secret_id = secret_id
        self.bucket = bucket
        self._init_oss()
        # image about
        self.list_file = list_file
        # data scale about
        self.rank = rank
        self.num_replicas = num_replicas
        self._collect_data()

    def _init_oss(self):
        auth = oss.Auth(self.key_id, self.secret_id)
        try:
            self.bucket = oss.Bucket(auth, self.endpoint, self.bucket, connect_timeout=300)
        except:
            raise ValueError("Oss server is unreachable!")

    def _collect_data(self):
        lines = []
        for _ in range(5):
            try:
                for l in str(self.bucket.get_object(self.list_file).read(), 'utf-8').split('\n'):
                    if l: lines.append(l)
                break
            except:
                continue

        self.lines = lines
        self.set_epoch(0)

    def set_epoch(self, epoch):
        g = torch.Generator()
        g.manual_seed(epoch)
        indices = torch.randperm(len(self.lines), generator=g).tolist()

        num_samples = int(math.ceil(len(self.lines) * 1.0 / self.num_replicas))
        total_size = num_samples * self.num_replicas

        indices += indices[:(total_size - len(indices))]
        assert len(indices) == total_size

        # subsample
        indices = indices[self.rank:total_size:self.num_replicas]
        assert len(indices) == num_samples

        print(f'Epoch: {epoch}, rank: {self.rank}, samples number: {len(indices)}')

        pairs = []
        for idx in indices:
            file_path, label = self.lines[idx].strip().split('#;#')
            pairs.append((file_path, int(label)))
        assert len(pairs) == num_samples

        self.pairs = pairs

    def __getitem__(self, index):
        file_path, label = self.pairs[index]

        for _ in range(10):
            try:
                image = Image.open(self.bucket.get_object(file_path)).convert('RGB')
                break
            except:
                print(f'oss: {file_path} failed, tried again')

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.pairs)

class OssImageLoader():
    def __init__(self, config):
        self.endpoint = config['data']['endpoint']
        self.key_id = config['data']['key_id']
        self.secret_id = config['data']['secret_id']
        self.bucket = config['data']['bucket']
        self.train_list = config['data']['train_list']
        self.val_list = config['data']['val_list']
        self.num_replicas = config['world_size']
        self.rank = config['rank']
        self.resize_size = config['data']['resize_size']
        self.data_workers = config['data']['data_workers']
        self.dual_views = config['data']['dual_views']

    def get_loader(self, stage, batch_size):
        self.get_dataset(stage)

        data_loader = torch.utils.data.DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=(stage not in ('val', 'test')),
            num_workers=self.data_workers,
            pin_memory=True,
            drop_last=True
        )
        return data_loader

    def get_dataset(self, stage):
        if stage in ('train', 'ft'):
            list_file = self.train_list
        else:
            list_file = self.val_list
        transform1 = get_transform(stage)
        if self.dual_views:
            transform2 = get_transform(stage, gb_prob=0.1, solarize_prob=0.2)
            transform = MultiViewDataInjector([transform1, transform2])
        else:
            transform = transform1
        self.dataset = OssImageDataset(
            transform, self.endpoint, self.key_id, self.secret_id, self.bucket,
            list_file, self.rank, self.num_replicas)

    def set_epoch(self, epoch):
        self.dataset.set_epoch(epoch)
