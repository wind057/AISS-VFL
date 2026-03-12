# This repository contains the code used for the experiments in the paper: 
# EF-VFL: Communication-efficient Vertical Federated Learning via Compressed Error Feedback(https://arxiv.org/abs/2406.14420).
# by Pedro Valdeira, João Xavier, Cláudia Soares, Yuejie Chi.

import numpy as np
import logging
import torch
import torch.nn as nn

from algorithm.aissvfl import Scale_Compressor
from source.data_process import split_cifar10, split_mnist
from source.party_class import Active_Party, Passive_Party
from source.utils import get_acc
from source.utils import communication_cost_counter, set_seed


class EFCompressor(nn.Module):
    
    def __init__(self, direct_compressor, shape):
        super().__init__()
        self.direct_compressor = direct_compressor
        self.shape = shape
        self.state = None
        self.register_full_backward_hook(self._backward_hook)

    def forward(self, x, indices, epoch):
        if self.state is None:
            self.state = torch.zeros(*self.shape, requires_grad=False, device=x.device)

        state_detached = self.state.detach()
        updated_state = state_detached.clone()
        if epoch == 0:
            updated_state[indices] = self.direct_compressor(x)
        else:
            updated_state[indices] = state_detached[indices] + self.direct_compressor(x - state_detached[indices])
        self.state = updated_state.detach()

        return updated_state[indices]

    def _backward_hook(self, module, grad_input, grad_output):
        return (grad_output[0], None, None)


class TopKCompressor(nn.Module):
    
    def __init__(self, compression_ratio):
        super().__init__()
        self.compression_ratio = compression_ratio
        self.register_full_backward_hook(self._backward_hook)

    def forward(self, x):
        k = max(1, int(x.numel() * self.compression_ratio))
        abs_x = x.abs().view(-1)
        _, topk_indices = torch.topk(abs_x, k)
        mask = torch.zeros_like(abs_x)
        mask[topk_indices] = 1.0
        compressed_x = (x.view(-1) * mask).view_as(x)
        return compressed_x
    
    def _backward_hook(self, module, grad_input, grad_output):
        return (grad_output[0],)


class QSGDCompressor(nn.Module):
    '''A biased (normalized) version of the QSGD compressor'''

    def __init__(self, n_bits):
        super().__init__()
        if not isinstance(n_bits, int) or n_bits < 1:
            raise ValueError('n_bits must be an integer >= 1')
        n_quantization_levels = 2 ** n_bits
        self.s = n_quantization_levels - 1
        self.register_full_backward_hook(self._backward_hook)
    
    def forward(self, x):
        tau = 1 + torch.min(torch.tensor(x.numel() / self.s ** 2), torch.sqrt(torch.tensor(x.numel())) / self.s)
        x_norm = torch.norm(x)
        x_in_quant_interval = self.s * torch.abs(x) / x_norm
        xi = torch.floor(x_in_quant_interval + torch.rand_like(x_in_quant_interval))
        return torch.sign(x) * x_norm * xi / (self.s * tau)
    
    def _backward_hook(self, module, grad_input, grad_output):
        return (grad_output[0],)


compressors_d = {"topk": TopKCompressor, "qsgd": QSGDCompressor}


class CompressionModule(nn.Module):
    def __init__(self, compressor=None, compression_parameter=None, compression_type=None, num_samples=None, cut_size=None):
        super().__init__()
        self.compressor = compressor
        self.compression_parameter = compression_parameter
        self.compression_type = compression_type
        self.num_samples = num_samples
        self.cut_size = cut_size

        if compressor is None:
            self.compression_layer = None
        else:
            if compression_parameter is None:
                raise ValueError("compression_parameter must be provided when a compressor is.")
            elif compression_type is None:
                raise ValueError("compression_type must be provided when a compressor is.")
            elif compression_type == "direct":
                self.compression_layer = compressors_d[compressor](compression_parameter)
            elif compression_type == "ef":
                self.compression_layer = EFCompressor(compressors_d[compressor](compression_parameter), (num_samples, cut_size))

    def forward(self, x, apply_compression=False, indices=None, epoch=None):
        if apply_compression and self.compression_layer is not None:
            if self.compression_type == "direct":
                x = self.compression_layer(x)
            elif self.compression_type == "ef":
                x = self.compression_layer(x, indices, epoch)
        return x


def efvfl(dataset, learning_rate, epochs, embed_dim, client_num, train_loader, test_loader, seed, device, Q, c_factor=None, debug_flag=True):
    set_seed(seed, device, debug=debug_flag)
    cost_counter = communication_cost_counter()

    if dataset == 'cifar10':
        pp_pool = [Passive_Party('resnet',
                                 in_channel=3,
                                 img_size=16,
                                 embed_dim=embed_dim) for i in range(client_num)]
    else:
        pp_pool = [Passive_Party('cnn',
                            in_channel=1,
                            img_size=14,
                            embed_dim=embed_dim) for i in range(client_num)]
    ap = Active_Party(embed_dim*client_num, 10)


    for lm in pp_pool:
        lm.model = lm.model.to(device)
        lm.set_optimizer(learning_rate)
    ap.model = ap.model.to(device)
    ap.set_optimizer(learning_rate)
    criterion = nn.CrossEntropyLoss().to(device)

    # train
    epoch_info_list = []

    if c_factor is not None:
        c_level = c_factor / 32
        compressor = Scale_Compressor(bit=c_factor)
        compressor_ef = CompressionModule(
            compressor='qsgd',
            compression_parameter=c_factor,  # n_bits
            compression_type='ef',
            num_samples=len(train_loader.dataset),
            cut_size=embed_dim
        )
    else:
        c_level = 1

    per_embed_size = embed_dim * 4 * c_level

    for epoch in range(epochs):
        embed_view = [None] * client_num
        loss_list = []
        for _x, _y, _idx in train_loader:
            _x = _x.to(device)
            _y = _y.to(device)
            if(dataset == 'cifar10'):
                sub_x_train = split_cifar10(_x, client_num)
            else:
                sub_x_train = split_mnist(_x, client_num)

            # init embed view
            # pp forward pass
            with torch.no_grad():
                for pp_idx, pp in enumerate(pp_pool):
                    _x = sub_x_train[pp_idx]
                    pp_output = pp.model(_x)
                    if c_factor is not None:
                        pp_output = compressor_ef(pp_output, apply_compression=True, indices=_idx, epoch=epoch)
                    embed_view[pp_idx] = pp_output
                    cost_counter.count_forward(per_embed_size * _x.shape[0])

            for q in range(Q):
                # each pp iteration
                for pp_idx, pp in enumerate(pp_pool):
                    _x = sub_x_train[pp_idx]
                    pp_out = pp.model(_x)
                    if epoch != 0:
                        if c_factor is not None:
                            pp_out = compressor_ef(pp_out, apply_compression=True, indices=_idx, epoch=epoch)
                        embed_view[pp_idx] = pp_out
                        cost_counter.count_forward(per_embed_size * _x.shape[0])
                    embed = torch.cat(embed_view, dim=1)
                    embed = embed.detach().requires_grad_()

                    ap_out = ap.model(embed)
                    loss = criterion(ap_out, _y)
                    loss_list.append(loss.item())
                    ap.optimizer.zero_grad()
                    loss.backward()
                    ap.update_model()
                    embed_g = embed.grad
                    if c_factor is not None:
                        embed_g, low, high = compressor.compress(embed_g, device)
                        embed_g = compressor.decompress(embed_g, low, high)
                    embed_grad_list = torch.chunk(embed_g, client_num, dim=1)

                    cost_counter.count_backward(per_embed_size * embed_g.shape[0])
                    pp.optimizer.zero_grad()
                    pp_out.backward(embed_grad_list[pp_idx])
                    pp.update_model()

        # test
        loss_mean = np.mean(loss_list)
        acc = get_acc(pp_pool, ap, test_loader, dataset, device, client_num)
        epoch_info = f'epoch: {epoch}, dataset: {dataset}, seed: {seed}, c_level: {c_level}, loss: {loss_mean:.5f}, acc: {acc:.5f}'
        epoch_info_list.append(epoch_info)
        logging.info(epoch_info)

    overall_cost = cost_counter.get_cost(unit='mb')
    return_info = f'dataset: {dataset}, seed: {seed}, c_level: {c_level}, overall_cost: {overall_cost:.5f}, acc: {acc:.5f}'
    return return_info, epoch_info_list
