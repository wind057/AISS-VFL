from algorithm.aissvfl import Scale_Compressor
from source.data_process import split_cifar10, split_mnist
from source.party_class import Active_Party, Passive_Party
from source.train import get_acc
from source.utils import communication_cost_counter, set_seed


import numpy as np
import torch
import torch.nn as nn


import logging


def fedbcd(dataset, learning_rate, epochs, embed_dim, client_num, train_loader, test_loader, seed, device, Q, c_factor=None, debug_flag=True):
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
    else:
        c_level = 1

    per_embed_size = embed_dim * 4 * c_level

    for epoch in range(epochs):
        embed_view = [None] * client_num
        loss_list = []
        for _x, _y in train_loader:
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
                    embed_view[pp_idx] = pp_output
                    cost_counter.count_forward(per_embed_size * _x.shape[0])

            for q in range(Q):
                # each pp iteration
                for pp_idx, pp in enumerate(pp_pool):
                    _x = sub_x_train[pp_idx]
                    pp_out = pp.model(_x)
                    if epoch != 0:
                        embed_view[pp_idx] = pp_out
                        cost_counter.count_forward(per_embed_size * _x.shape[0])
                    embed = torch.cat(embed_view, dim=1)
                    if c_factor is not None:
                        embed, low, high = compressor.compress(embed, device)
                        embed = compressor.decompress(embed, low, high)
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