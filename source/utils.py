import csv
import random
import re
import torch
import numpy as np
import logging
from pathlib import Path
import os
import copy

def setupLogger(filename, record=True):
    # setup logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(filename)s %(funcName)s [line:%(lineno)d] %(levelname)s %(message)s')

    # setup file handler
    if record:
        file_path = Path(filename)
        if not file_path.parent.exists():
            os.makedirs(file_path.parent)

        file_path.touch(exist_ok=True)

        fh = logging.FileHandler(file_path, encoding='utf8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    logging.info('Start print log......')
    return logger


def get_available_folder(base_folder):
    base_path = Path(base_folder)
    counter = 1

    while base_path.exists():
        base_path = Path(f"{base_folder}_{counter}")
        counter += 1

    return base_path


def set_seed(seed, device, debug=True):
    if(seed >= 0):
        if(device == 'cuda'):
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
            if debug:
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
            else: # faster but less reproducibility
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
        else:
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

def parse_line(line):
    fields = re.findall(r"(\w+):\s*([0-9\.a-zA-Z]+)", line)
    return {key: value for key, value in fields}

def dump_data(raw_data, outfile_path, mode='a'):
    out_dir = os.path.dirname(outfile_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    numbers = [parse_line(line) for line in raw_data]  
    csv_head = numbers[0].keys()
    with open(outfile_path, mode=mode, newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=csv_head)
        
        if file.tell() == 0:
            writer.writeheader()
        writer.writerows(numbers)

class communication_cost_counter():
    def __init__(self):
        self.forward_cost = 0
        self.backward_cost = 0
    
    def count_forward(self, data_size):
        self.forward_cost = self.forward_cost + data_size

    def count_backward(self, data_size):
        self.backward_cost = self.backward_cost + data_size

    def get_cost(self, unit='byte'):
        uniformed_cost = self.forward_cost + self.backward_cost
        if unit == 'kb':
            uniformed_cost = uniformed_cost / 1024
        elif unit == 'mb':
            uniformed_cost = uniformed_cost / 1024 / 1204

        return uniformed_cost


