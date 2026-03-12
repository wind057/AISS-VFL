import os
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
import pickle

from torchvision import datasets, transforms

def split_cifar10(x : torch.tensor, client_num=2, fill_value=0):
    split_list = []
    if(client_num == 1):
        split_list.append(x)
    elif(client_num == 2): # padding (n, 1, 16, 32) to (n, 1, 32, 32)
        sub_x = torch.chunk(x, client_num, dim=2)
        for _x in sub_x:
            _x_pad = F.pad(_x, (0, 0, 16, 0), mode='constant', value=fill_value)
            split_list.append(_x_pad)
    elif(client_num == 4):
        q1 = x[:, :, :16, :16]
        q2 = x[:, :, :16, 16:]
        q3 = x[:, :, 16:, :16]
        q4 = x[:, :, 16:, 16:]
        split_list = [q1, q2, q3, q4]
    return split_list

def split_mnist(x : torch.tensor, client_num, fill_value=0):
    split_list = []
    if(client_num == 2): # padding (n, 1, 14, 28) to (n, 1, 28, 28)
        sub_x = torch.chunk(x, client_num, dim=2)
        for _x in sub_x:
            _x_pad = F.pad(_x, (0, 0, 14, 0), mode='constant', value=fill_value)
            split_list.append(_x_pad)
    elif(client_num == 4):
        q1 = x[:, :, :14, :14]
        q2 = x[:, :, :14, 14:]
        q3 = x[:, :, 14:, :14]
        q4 = x[:, :, 14:, 14:]
        split_list = [q1, q2, q3, q4]
    return split_list


def load_mnist(csv_file, train=True):
    if train:
        csv_file = csv_file + 'train.csv'
    else:
        csv_file = csv_file + 'test.csv'
    data = pd.read_csv(csv_file)
    features = data.iloc[:, 1:].values.astype('float32') / 255
    features = features.reshape(-1, 1, 28, 28)
    label = data.iloc[:, 0].astype('int64')
    return torch.tensor(features), torch.tensor(label)

def load_cifar10(dir, train=True, transform=None):
    x = []
    y = []
    if train:
        for i in range(1, 6):
            batch_path = os.path.join(dir, f'data_batch_{i}')
            with open(batch_path, 'rb') as f:
                raw_data_dict = pickle.load(f, encoding='bytes')
                x_batch = raw_data_dict[b'data']
                y_batch = raw_data_dict[b'labels']

                x_chw = (x_batch.reshape(-1, 3, 32, 32).astype(np.float32))
                
                x.append(torch.from_numpy(x_chw))
                y.append(torch.tensor(y_batch))
        x = torch.vstack(x)
        y = torch.hstack(y)
    else:
        batch_path = os.path.join(dir, 'test_batch')
        with open(batch_path, 'rb') as f:
            raw_data_dict = pickle.load(f, encoding='bytes')
            x_batch = raw_data_dict[b'data']
            y_batch = raw_data_dict[b'labels']
            x_chw = (x_batch.reshape(-1, 3, 32, 32).astype(np.float32))
            x = torch.from_numpy(x_chw)
            y = torch.tensor(y_batch)
    return x, y


class IndexDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, y, idx

def load_dataset(dataset_dir):
    transform_mnist = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    transform_fmnist = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    transform_cifar10_trian = transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        ),
        transforms.RandomErasing(p=0.5, scale=(0.02,0.25), ratio=(0.33,3.3))
    ])
    transform_cifar10_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
        )
    ])

    # load dataset
    data = {
        'mnist' : (datasets.MNIST(root=dataset_dir, train=True, download=True, transform=transform_mnist),
                    datasets.MNIST(root=dataset_dir, train=False, download=True, transform=transform_mnist)),
        'fmnist' : (datasets.FashionMNIST(root=dataset_dir, train=True, download=True, transform=transform_fmnist),
                    datasets.FashionMNIST(root=dataset_dir, train=False, download=True, transform=transform_fmnist)),
        'cifar10' : (datasets.CIFAR10(root=dataset_dir, train=True, download=False, transform=transform_cifar10_trian),
                    datasets.CIFAR10(root=dataset_dir, train=False, download=False, transform=transform_cifar10_test)),
    }

    return data