import numpy as np
import torch
import copy
from torch.func import grad, vmap
import torch.nn as nn
import logging
from source.data_process import split_cifar10, split_mnist
from source.party_class import Active_Party, Passive_Party
from source.train import get_acc
from source.utils import communication_cost_counter, set_seed


def normalize(value_list):
    mean = torch.mean(value_list)
    std = torch.std(value_list)
    normalized_value = (value_list - mean) / std
    normalized_value = torch.sigmoid(normalized_value)
    
    return normalized_value

def select_sample(ap, embed, label, threshold):
    model_copy = copy.deepcopy(ap.model)
    embed = embed.detach().requires_grad_()

    def loss_fn(params, x, y):
        preds = torch.func.functional_call(model_copy, params, (x,))
        loss = nn.CrossEntropyLoss(reduction='none')(preds.unsqueeze(0), y.unsqueeze(0))
        return loss.squeeze()

    grad_fn = grad(loss_fn, argnums=(0,1))
    model_params = dict(model_copy.named_parameters()) 
    model_grads, embed_grad = vmap(grad_fn, in_dims=(None, 0, 0))(model_params, embed, label)

    model_grad_list = []
    for name, grads in model_grads.items():
        model_grad_list.append(grads.reshape(grads.shape[0], -1))
    model_grad = torch.cat(model_grad_list, dim=1)

    embed_gnorm = torch.norm(embed_grad, p=2, dim=1)
    model_gnorm = torch.norm(model_grad, p=2, dim=1)

    c = embed_gnorm + model_gnorm
    normalized_grad = normalize(c)
    idx_result = torch.where(normalized_grad > threshold)[0]
    selected_normalized_grad = torch.index_select(normalized_grad, dim=0, index=idx_result)

    return idx_result, selected_normalized_grad


# paper title: A Unified Solution for Privacy and Communication Efficiency in Vertical Federated Learning
# code repo: https://github.com/GanyuWang/VFL-CZOFO
class Scale_Compressor():
    def __init__(self, bit):
        self.compression_type = "Scale"
        self.bit = bit
        if bit <= 8:
            self.dtype = torch.uint8
        elif bit == 16:
            self.dtype = torch.uint16

    def compress(self, tensor, device):
        # input a tensor. 
        # high, low, bit. 
        tensor = tensor.contiguous()
        low = tensor.min().item()
        high = tensor.max().item()
        # boundaries = torch.tensor(np.linspace(low.cpu(), high.cpu(), 2**self.bit)).to(self.device)
        boundaries = torch.linspace(low, high, steps=2**self.bit, device=device, dtype=tensor.dtype)
        compressed_tensor = torch.bucketize(input = tensor, boundaries=boundaries)
        compressed = compressed_tensor.clone().detach().to(self.dtype)
        return compressed, low, high
    
    def decompress(self, compressed, low, high):
        # assert self.bit <= 8
        decomp = low + compressed*(high - low)/(2**self.bit)
        return decomp

    def comm_cost_in_bit(self, compressed):
        n_element = torch.numel(compressed)
        base = n_element * self.bit
        extra = 32 * 2 # float. low and high.  
        return base + extra


def aissvfl(dataset, learning_rate, epochs, embed_dim, client_num, train_loader, test_loader, seed, device, threshold, c_factor=None, debug_flag=True, rescale=True):
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
    criterion = nn.CrossEntropyLoss(reduction="none").to(device)

    # train
    epoch_info_list = []
    # initialize compressor
    if c_factor is not None:
        c_level = c_factor / 32
        compressor = Scale_Compressor(bit=c_factor)
    else:
        c_level = 1

    # embed_dim * client_num * 4bytes * comp_level
    per_embed_size = embed_dim * client_num * 4 * c_level

    for epoch in range(epochs):
        count_used_sample = 0
        count_all_sample = 0
        loss_list = []
        for _x, _y in train_loader:
            pp_output_list = []
            _x = _x.to(device)
            _y = _y.to(device)
            if(dataset == 'cifar10'):
                sub_x_train = split_cifar10(_x, client_num)
            else:
                sub_x_train = split_mnist(_x, client_num)

            for pp_idx, pp in enumerate(pp_pool):
                sub_x = sub_x_train[pp_idx]
                pp_output = pp.model(sub_x)
                pp_output_list.append(pp_output)
            embed = torch.cat(pp_output_list, dim=1)

            if c_factor is not None:
                embed, low, high = compressor.compress(embed, device)
                embed = compressor.decompress(embed, low, high)

            cost_counter.count_forward(per_embed_size * embed.shape[0])

            # sample selection
            selected_sample_index, selected_importance_score = select_sample(ap, embed, _y, threshold)
            count_all_sample = count_all_sample + _y.shape[0]

            if selected_sample_index.shape[0] == 0:
                continue
            count_used_sample = count_used_sample + selected_sample_index.shape[0]

            if rescale is True:
                rescale_weight = 1 / selected_importance_score
            else:
                rescale_weight = 1

            selected_embed = torch.index_select(embed, dim=0, index=selected_sample_index).detach().requires_grad_()
            selected_y = torch.index_select(_y, dim=0, index=selected_sample_index)
            ap_output = ap.model(selected_embed)
            ap.optimizer.zero_grad()
            losses = criterion(ap_output, selected_y)

            # rescaling
            loss = torch.mean(losses * rescale_weight)

            loss_list.append(loss.item())
            loss.backward()
            ap.update_model()

            cost_counter.count_backward(per_embed_size * selected_sample_index.shape[0])
            selected_g = selected_embed.grad

            if c_factor is not None:
                # compressed_selected_g = quantize_scalar(selected_g, quant_level=c_factor)
                compressed_selected_g, low, high = compressor.compress(selected_g, device)
                selected_g = compressor.decompress(compressed_selected_g, low, high)

            inputs_grad_list = torch.chunk(selected_g, client_num, dim=1)
            inputs_grad_list = [g.contiguous() for g in inputs_grad_list]
            pp_output_selected_list = [o[selected_sample_index] for o in pp_output_list]

            for pp_idx, pp in enumerate(pp_pool):
                pp_output = pp_output_selected_list[pp_idx]
                inputs_grad = inputs_grad_list[pp_idx]
                pp.optimizer.zero_grad()
                pp_output.backward(inputs_grad)
                pp.update_model()

        back_ratio = count_used_sample / count_all_sample
        if len(loss_list) == 0:
            loss_mean = -1
        else:
            loss_mean = np.mean(loss_list)
        # test
        acc = get_acc(pp_pool, ap, test_loader, dataset, device, client_num)
        epoch_info = f'epoch: {epoch}, dataset: {dataset}, seed: {seed}, tau: {threshold}, c_level: {c_level}, back_ratio: {back_ratio:.5f}, loss: {loss_mean:.5f}, acc: {acc:.5f}'
        epoch_info_list.append(epoch_info)
        logging.info(epoch_info)

    overall_cost = cost_counter.get_cost(unit='mb')
    return_info = f'dataset: {dataset}, seed: {seed}, c_level: {c_level}, tau: {threshold:.5f}, acc: {acc:.5f}, overall_cost: {overall_cost:.5f}'
    return return_info, epoch_info_list