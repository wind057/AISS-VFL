import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import logging
import numpy as np
import os
from algorithm.aissvfl import Scale_Compressor
from source.data_process import IndexDataset, split_cifar10, split_mnist
from source.utils import communication_cost_counter, set_seed

# Title: Practical Vertical Federated Learning With Unsupervised Representation Learning
# Author: Z. Wu, Q. Li and B. He

def generate_random_targets(n: int, z: int, method='sphere'):
    """
    Generate a matrix of random target assignment.
    Each target assignment vector has unit length (hence can be view as random point on hypersphere)
    :param n: the number of samples to generate.
    :param z: the latent space dimensionality
    :return: the sampled representations
    """
    if method == 'sphere':
        # Generate random targets using gaussian distrib.
        samples = np.random.normal(0, 1, (n, z)).astype(np.float32)
        # rescale such that fit on unit sphere.
        radiuses = np.expand_dims(np.sqrt(np.sum(np.square(samples), axis=1)), 1)
        # return rescaled targets
        return samples / radiuses
    elif method == 'uniform':
        return np.random.uniform(0, 1, (n, z)).astype(np.float32)
    else:
        print("Unsupported random method")
        return None

def calc_optimal_target_permutation(feats: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Compute the new target assignment that minimises the SSE between the mini-batch feature space and the targets.
    :param feats: the learnt features (given some input images)
    :param targets: the currently assigned targets.
    :return: the targets reassigned such that the SSE between features and targets is minimised for the batch.
    """
    # Compute cost matrix
    cost_matrix = np.zeros([feats.shape[0], targets.shape[0]])
    # calc SSE between all features and targets
    for i in range(feats.shape[0]):
         cost_matrix[:, i] = np.sum(np.square(feats-targets[i, :]), axis=1)

    _, col_ind = linear_sum_assignment(cost_matrix)
    # Permute the targets based on hungarian algorithm optimisation
    targets[range(feats.shape[0])] = targets[col_ind]
    return targets

def make_layer(block, in_channels, out_channels, num_blocks, stride):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
        layers.append(block(in_channels, out_channels, stride))
        in_channels = out_channels
    return nn.Sequential(*layers)

class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out

class LeNet_nat(nn.Module):
    def __init__(self, in_channel, img_size, z_dim, model_type=None):
        super(LeNet_nat, self).__init__()
        self.in_channel = in_channel
        self.img_size = img_size
        act = nn.ReLU
        if model_type == 'resnet-18':
                self.body = nn.Sequential(
                nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                make_layer(ResBlock, 64, 64, 2, stride=1),
                make_layer(ResBlock, 64, 128, 2, stride=2),
                make_layer(ResBlock, 128, 256, 2, stride=2),
                make_layer(ResBlock, 256, 512, 2, stride=2),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(1)
            )
        else:
            self.body = nn.Sequential(
                nn.Conv2d(in_channel, 12, kernel_size=3, stride=1),
                act(),
                nn.Conv2d(12, 12, kernel_size=3, stride=1),
                act(),
                nn.MaxPool2d(kernel_size=2),
                nn.Flatten(1)
            )
            
        self.fc1 = nn.Linear(self.fc_input_size, z_dim)
    
    def forward(self, x):
        out = self.body(x)
        # out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out
    
    @property
    def fc_input_size(self):
        x = torch.randn((1, self.in_channel, self.img_size, self.img_size))
        out = self.body(x)
        return out.shape[1]

class IndexDataset_oneshot(Dataset):
    def __init__(self, dataset, fake_label=None, embed=None):
        self.dataset = dataset
        self.fake_label = fake_label
        self.embed = embed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if self.fake_label is not None:
            y = self.fake_label[idx]
        if self.embed is not None:
            x = self.embed[idx]
        return x, y, idx
    def update_fake_label(self, idx, value):
        self.fake_label[idx] = value

class Passive_Party_Oneshot():
    def __init__(self, dataset_type, in_channel=1, img_size=14, z_dim=128):
        self.dataset_type = dataset_type
        if dataset_type == 'cifar10':
            # model_type = 'resnet-18'
            model_type = 'cnn'
        else:
            model_type = 'cnn'
        self.model = LeNet_nat(in_channel=in_channel, img_size=img_size, z_dim=z_dim, model_type=model_type)
        self.embed_list = None

    def save_model(self, dir_path, pp_idx=None):
        file_path = dir_path + '{}-pp{}.pth'.format(self.dataset_type, pp_idx)
        model = self.model.cpu()
        torch.save(model.state_dict(), file_path)

    def load_model(self, dir_path, pp_idx=None, device=None):
        file_path = dir_path + '{}-pp{}.pth'.format(self.dataset_type, pp_idx)
        self.model.load_state_dict(torch.load(file_path, weights_only=True))
        if device is not None:
            self.model.to(device)

    @torch.no_grad()
    def encode(self, x, device):
        self.model = self.model.to(device)
        output = self.model(x)
        return output

class FC(nn.Module):
    def __init__(self, in_dim=512, class_num=10):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_dim, class_num)

    def forward(self, x):
        out = self.fc(x)
        return out

class Active_Party_Oneshot:
    def __init__(self, class_num=10, z_dim=512):

        self.model = FC(in_dim=z_dim, class_num=class_num)

    def train_active(self, runConfig, dataset_type, seed, c_level, trainset, testset, device):
        # algorithm: VFLOnce, FC, TableNet, XGBoost
        batch_size = runConfig['batch_size']
        epochs = runConfig['epochs']
        learning_rate = runConfig['learning_rate']

        self.model = self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss().to(device)
        train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(dataset=testset, batch_size=1000, shuffle=False, num_workers=2)
        epoch_info_list = []
        for epoch in range(epochs):
            loss_list = []
            for _x, _y, _ in train_loader:
                _x = _x.to(device)
                _y = _y.to(device)
                optimizer.zero_grad()
                _output = self.model(_x)
                loss = criterion(_output, _y)
                loss_list.append(loss.item())
                loss.backward()
                optimizer.step()
            # testing
            with torch.no_grad():
                label_list = []
                pred_list = []
                self.model.eval()
                for x_t, y_t, _ in test_loader:
                    x_t, y_t = x_t.to(device), y_t.to(device)
                    y_pred = self.model(x_t)
                    label_list.append(y_t)
                    y_pred = torch.argmax(y_pred, dim=1)
                    pred_list.append(y_pred)
                self.model.train()
                all_label = torch.cat(label_list)
                all_pred = torch.cat(pred_list)
                correct = (all_label == all_pred).sum()
                acc = correct / all_label.shape[0]
            loss_mean = np.mean(loss_list)
            
            epoch_info = f'epoch: {epoch}, dataset: {dataset_type}, seed: {seed}, c_level: {c_level}, loss: {loss_mean:.5f}, acc: {acc:.5f}'
            epoch_info_list.append(epoch_info)
            logging.info(epoch_info)
        return acc, epoch_info_list

def train_passive(dataset_type, pp_pool, runConfig, trainset, embed_dim, client_num, device):
    learning_rate = runConfig['learning_rate']
    epochs = runConfig['epochs']
    
    batch_size = runConfig['batch_size']
    weight_decay = runConfig['weight_decay']

    fake_label = generate_random_targets(len(trainset), embed_dim)
    trainset = IndexDataset_oneshot(trainset, fake_label=fake_label)
        
    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    criterion = nn.MSELoss().to(device)
    optimizer_list = [optim.Adam(pp.model.parameters(), lr=learning_rate, weight_decay=weight_decay) for pp in pp_pool]

    for epoch in range(epochs):
        loss_list = [[] for i in range(client_num)]
        for x, y, sample_idx in train_loader:
            if(dataset_type == 'cifar10'):
                sub_x_train = split_cifar10(x, client_num)
            else:
                sub_x_train = split_mnist(x, client_num)


            for pp_idx, pp in enumerate(pp_pool):
                
                model = pp.model.to(device)
                optimizer = optimizer_list[pp_idx]
                _x, _y = sub_x_train[pp_idx], y

                update_freq = runConfig.get('update_freq')

                _x = _x.to(device)
                _y = _y.to(device)
                optimizer.zero_grad()
                output = model(_x)
                if((epoch + 1) % update_freq == 0):
                    # update P with hungarian algorithm
                    output_np = output.cpu().detach().numpy()
                    new_targets = calc_optimal_target_permutation(output_np, _y.cpu().detach().numpy())
                    new_targets_tensor = torch.from_numpy(new_targets)
                    trainset.update_fake_label(sample_idx, new_targets_tensor) # 更新标签
                    _y = new_targets_tensor.to(device)
                loss = criterion(output, _y)
                loss_list[pp_idx].append(loss.item())
                loss.backward()
                optimizer.step()

        for pp_idx, pp in enumerate(pp_pool):

            loss_mean = np.mean(loss_list[pp_idx])
            logging.info(f"Passvie Party: {pp_idx}, Train Epoch: {epoch}, Loss: {loss_mean:.6f}")

def get_embed(pp_pool, dataset, dataset_type, device):
    client_num = len(pp_pool)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False, num_workers=2)

    embed_list = []
    for x, _ in dataloader:
        x = x.to(device)
        if(dataset_type == 'cifar10'):
            sub_x_train = split_cifar10(x, client_num)
        else:
            sub_x_train = split_mnist(x, client_num)
        output_list = []
        for pp_idx, pp in enumerate(pp_pool):
            output = pp.encode(sub_x_train[pp_idx], device)
            output_list.append(output)
        embed = torch.cat(output_list, dim=1)
        embed_list.append(embed)

    return torch.cat(embed_list).cpu()


def fedonce(dataset_type, runConfig, trainset, testset, embed_dim, cache_flag, client_num, seed, device, c_factor=None, debug_flag=True):
    set_seed(seed, device, debug=debug_flag)
    if dataset_type == 'cifar10':
        pp_pool = [Passive_Party_Oneshot(dataset_type, in_channel=3, img_size=16, z_dim=embed_dim) for i in range(client_num)]
    else:
        pp_pool = [Passive_Party_Oneshot(dataset_type, in_channel=1, img_size=14, z_dim=embed_dim) for i in range(client_num)]
    ap = Active_Party_Oneshot(class_num=10,
                        z_dim=embed_dim*client_num)
    cost_counter = communication_cost_counter()

    if c_factor is not None:
        c_level = c_factor / 32
        compressor = Scale_Compressor(bit=c_factor)
    else:
        c_level = 1

    per_embed_size = embed_dim * client_num * 4 * c_level

    exist_flag = False
    for cache_name in os.listdir('./checkpoint/'):
        _dataset_name = cache_name.split('-')[0]
        if dataset_type == _dataset_name:
            exist_flag = True
            break
    # pp train
    if cache_flag and exist_flag:
        logging.info('Loading passive party')
        for _pp_idx, _pp in enumerate(pp_pool):
            _pp.load_model('./checkpoint/', pp_idx=_pp_idx, device=device)
    else:
        logging.info('Training passive party ')
        train_passive(dataset_type, pp_pool, runConfig['passive'][dataset_type], trainset, embed_dim, client_num, device)
        for pp_idx, pp in enumerate(pp_pool):
            pp.save_model(dir_path='./checkpoint/', pp_idx=pp_idx)

        logging.info('Training passive party complete')
        logging.info('Training active party')

    # ap train
    embed_train = get_embed(pp_pool, trainset, dataset_type, device=device)
    embed_test = get_embed(pp_pool, testset, dataset_type, device=device)

    if c_factor is not None:
        embed_train, low, high = compressor.compress(embed_train, device='cpu')
        embed_train = compressor.decompress(embed_train, low, high)
        embed_test, low, high = compressor.compress(embed_test, device='cpu')
        embed_test = compressor.decompress(embed_test, low, high)
    cost_counter.count_forward(per_embed_size)

    last_acc, epoch_info_list = ap.train_active(runConfig['active'][dataset_type], dataset_type, seed, c_level, IndexDataset_oneshot(trainset, embed=embed_train), IndexDataset_oneshot(testset, embed=embed_test), device)

    overall_cost = cost_counter.get_cost(unit='mb')
    return_info = f'dataset: {dataset_type}, seed: {seed}, c_level: {c_level}, overall_cost: {overall_cost:.5f}, acc: {last_acc:.5f}'

    return return_info, epoch_info_list