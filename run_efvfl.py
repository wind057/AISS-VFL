from itertools import product
import os
from source.data_process import IndexDataset, load_dataset
from algorithm.efvfl import efvfl
from source.utils import dump_data, setupLogger
import toml
from tqdm import tqdm
import torch

if __name__ == '__main__':
    config = toml.load('./config/config.toml')
    device = config['device']
    client_num = config['client_num']
    embed_dim = config['embed_dim']
    seeds = config['seeds']  
    dataset_list = config['dataset_list']
    dataset_dir = config['dataset_path']
    work_dir = config['work_dir']
    debug_flag = config['debug_flag']

    # task
    task_config = toml.load('./config/efvfl.toml')
    k_list = task_config['k_list']
    Q = task_config['Q']

    
    os.chdir(work_dir)

    
    log_path = './train_log/log_efvfl.txt'
    result_path = './output/result_efvfl.csv'
    epoch_path = './output/epoch_efvfl.csv'

    
    if os.path.exists(result_path):
        os.remove(result_path)
    
    if os.path.exists(log_path):
        os.remove(log_path)
    
    if os.path.exists(epoch_path):
        os.remove(epoch_path)
        

    logger = setupLogger(log_path)

    data = load_dataset(dataset_dir)

    b_dict = {
        0.125: 4,
        0.25: 8,
        None:None
    }

    for dataset in dataset_list:
        result_list = []
        epoch_list = []
        train_config = task_config[dataset]
        lr = train_config['lr']
        epochs = train_config['epochs']
        batch_size = train_config['batch_size']
        train_loader = torch.utils.data.DataLoader(IndexDataset(data[dataset][0]), batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
        test_loader = torch.utils.data.DataLoader(data[dataset][1], batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)
        
        for seed, k in tqdm(product(seeds, k_list), total=len(k_list) * len(seeds), desc=dataset):
            if k == 1.0:
                k = None
            result_info, epoch_info_list = efvfl(dataset, lr, epochs, embed_dim, client_num, train_loader, test_loader, seed, device, Q, c_factor=b_dict[k], debug_flag=debug_flag)
            result_list.append(result_info)
            epoch_list.extend(epoch_info_list)
        dump_data(result_list, result_path)
        dump_data(epoch_list, epoch_path)
    print("All tasks completed.")


