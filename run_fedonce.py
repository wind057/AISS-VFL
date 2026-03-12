from itertools import product
from tqdm import tqdm
from source.data_process import load_dataset
from algorithm.fedonce import fedonce
from source.utils import dump_data, setupLogger
import toml
import os

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
    task_config = toml.load('./config/fedonce.toml')
    k_list = task_config['k_list']
    cache_flag = task_config['cache_flag']

    
    os.chdir(work_dir)

    
    log_path = './train_log/log_fedonce.txt'
    result_path = './output/result_fedonce.csv'
    epoch_path = './output/epoch_fedonce.csv'

    
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

    for dataset_type in dataset_list:
        result_list = []
        epoch_list = []
        trainset = data[dataset_type][0]
        testset = data[dataset_type][1]

        for seed, k in tqdm(product(seeds, k_list), total=len(k_list) * len(seeds), desc=dataset_type):
            if k == 1.0:
                k = None

            result_info, epoch_info_list = fedonce(dataset_type, task_config, trainset, testset, embed_dim, cache_flag, client_num, seed, device, c_factor=b_dict[k], debug_flag=debug_flag)
            result_list.append(result_info)
            epoch_list.extend(epoch_info_list)
        dump_data(result_list, result_path)
        dump_data(epoch_list, epoch_path)
    print("All tasks completed.")