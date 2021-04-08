import os
import pandas as pd
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

DATASETS = ['cub', 'awa2']


class DataLoader(object):
    def __init__(self, dataset='awa2'):
        assert dataset in DATASETS
        self.dataset = dataset

    def load_data(self, batch_size=128, num_workers=8, root='./data/'):

        if self.dataset == 'cub':
            dataset_class = CUB
            n_classes = 200

        elif self.dataset == 'awa2':
            dataset_class = AWA2
            n_classes = 50

        transform_train = transforms.Compose([transforms.ToPILImage(),
                                              transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                   std=(0.229, 0.224, 0.225))])

        transform_test = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize(224),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                  std=(0.229, 0.224, 0.225))])

        train_dataset = dataset_class(root=root,
                                      train=True,
                                      transform=transform_train,
                                      download=True)

        test_dataset = dataset_class(root=root,
                                     train=False,
                                     transform=transform_test)

        val_size = int(len(train_dataset) * 0.1)
        train_size = len(train_dataset) - val_size

        train_dataset, val_dataset = torch.utils.data.dataset.random_split(train_dataset, [train_size, val_size])

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=num_workers)

        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=num_workers)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=num_workers)

        dataloaders = {'train': train_loader,
                       'val': val_loader,
                       'test': test_loader}

        return dataloaders, n_classes


class CUB(Dataset):
    """CUB200-2011 dataset."""
    attribute_file = 'attributes/class_attribute_labels_continuous.txt'

    def __init__(self, root, train=True, transform=None, normalize=True,
                 download=None):
        self.root = os.path.join(root, 'cub')
        self.train = train
        self.transform = transform
        self.data_dir = os.path.join(self.root, 'images')

        train_test_split = pd.read_csv(os.path.join(self.root, 'train_test_split.txt'),
                                       sep=' ', index_col=0, header=None)
        if train:
            is_train_image = 1
        else:
            is_train_image = 0
        self.img_ids = train_test_split[train_test_split[1] == is_train_image].index.tolist()
        self.id_to_img = pd.read_csv(os.path.join(self.root, 'images.txt'),
                                     sep=' ', index_col=0, header=None)

        raw_mtx = np.loadtxt(os.path.join(self.root,
                                          self.attribute_file))
        raw_mtx[raw_mtx == -1] = 0
        raw_mtx = raw_mtx / raw_mtx.max()
        self.attribute_mtx = torch.tensor(raw_mtx, dtype=torch.float)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_name = self.id_to_img[self.id_to_img.index == img_id].values[0][0]
        img_path = os.path.join(self.data_dir, img_name)

        img = imageio.imread(img_path, pilmode='RGB')
        label = int(img_name[:3]) - 1

        if self.transform:
            img = self.transform(img)

        return img, label, img_path


class AWA2(Dataset):
    """Animals with Attributes 2 dataset."""
    split_file = 'train_test_classification_split.txt'
    data_dir = 'awa2'
    attribute_file = 'predicate-matrix-continuous.txt'

    def __init__(self, root, train=True, transform=None, normalize=True,
                 download=None):
        self.root = os.path.join(root, self.data_dir)
        self.train = train
        self.transform = transform

        meta_data = pd.read_csv(os.path.join(self.root,
                                             self.split_file),
                                sep=' ', index_col=0, header=None)
        if train:
            is_train_image = 1
        else:
            is_train_image = 0
        self.img_ids = meta_data[meta_data[3] == is_train_image].index.tolist()
        self.id_to_img = meta_data

        raw_mtx = np.loadtxt(os.path.join(self.root,
                                          self.attribute_file))
        raw_mtx[raw_mtx == -1] = 0
        raw_mtx = raw_mtx / raw_mtx.max()
        self.attribute_mtx = torch.tensor(raw_mtx, dtype=torch.float)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_meta_data = self.id_to_img[self.id_to_img.index == img_id]
        img_name = img_meta_data.values[0][0]
        img_path = os.path.join(self.root, img_name)

        img = imageio.imread(img_path, pilmode='RGB')
        label = img_meta_data.values[0][1] - 1

        if self.transform:
            img = self.transform(img)

        return img, label
