import torch
import logging
import torch.nn as nn
from algorithm.aissvfl import Scale_Compressor
from algorithm.compressor import topk
from source.data_process import split_cifar10, split_mnist
from source.party_class import Active_Party, Passive_Party
from source.utils import communication_cost_counter, set_seed
import numpy as np

def vfl(dataset, learning_rate, epochs, embed_dim, client_num, train_loader, test_loader, seed, device, c_type=None, c_factor=None, debug_flag=True):
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

    if c_factor is not None:
        compressor = Scale_Compressor(bit=c_factor)

    epoch_info_list = []
    for epoch in range(epochs):
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
            if c_factor is None:
                c_type = 'null'
                c_level = 1
            elif c_type == 'topk':
                embed = topk(embed, c_factor)
                c_level = c_factor
            elif c_type == 'scalar':
                c_level = c_factor / 32
                embed, low, high = compressor.compress(embed, device)
                embed = compressor.decompress(embed, low, high)

            per_embed_size = embed_dim * client_num * 4 * c_level
            cost_counter.count_forward(per_embed_size * embed.shape[0])

            embed = embed.detach().requires_grad_()
            _y = _y.to(device)
            ap_output = ap.model(embed)
            ap.optimizer.zero_grad()
            loss = criterion(ap_output, _y)
            loss_list.append(loss.item())
            loss.backward()
            ap.update_model()

            embed_grad = embed.grad.clone().detach()
            cost_counter.count_backward(per_embed_size * embed_grad.shape[0])

            if c_factor is None:
                pass
            elif c_type == 'topk':
                embed_grad = topk(embed_grad, c_factor)
            elif c_type == 'scalar':
                embed_grad, low, high = compressor.compress(embed_grad, device)
                embed_grad = compressor.decompress(embed_grad, low, high)

            inputs_grad_list = torch.chunk(embed_grad, client_num, dim=1)

            for pp_idx, pp in enumerate(pp_pool):
                inputs_grad = inputs_grad_list[pp_idx]
                pp.optimizer.zero_grad()
                pp_output = pp_output_list[pp_idx]
                pp_output.backward(inputs_grad)
                pp.update_model()

        # test
        if len(loss_list) == 0:
            loss_mean = -1
        else:
            loss_mean = np.mean(loss_list) 
        acc = get_acc(pp_pool, ap, test_loader, dataset, device, client_num)
        epoch_info = f'epoch: {epoch}, dataset: {dataset}, seed: {seed}, compressor: {c_type}, c_level: {c_level}, loss: {loss_mean:.5f}, acc: {acc:.5f}'
        epoch_info_list.append(epoch_info)
        logging.info(epoch_info)
    
    overall_cost = cost_counter.get_cost(unit='mb')
    result_info = f'dataset: {dataset}, seed: {seed}, compressor: {c_type}, c_level: {c_level}, overall_cost: {overall_cost:.5f}, acc: {acc:.5f}'
    return result_info, epoch_info_list


def get_acc(pp_pool, ap, test_loader, dataset, device, client_num=4):
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        ap.model.eval()
        for pp in pp_pool:
            pp.model.eval()
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                embed_list = []

                if(dataset == 'cifar10'):
                    x_list = split_cifar10(x, client_num)
                else:
                    x_list = split_mnist(x, client_num)

                for i, pp in enumerate(pp_pool):
                    pp_output = pp.model(x_list[i])
                    embed_list.append(pp_output)

                embed = torch.cat(embed_list, dim=1)
                output = ap.model(embed)
                pred = torch.argmax(output, dim=1)
                all_predictions.append(pred)
                all_labels.append(y)

    ap.model.train()
    for pp in pp_pool:
        pp.model.train()

    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    correct = (all_predictions == all_labels).sum()
    total = len(all_labels)
    accuracy = correct / total

    return accuracy
