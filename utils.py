import os
import os.path as osp
import random

import numpy as np
import torch
import winsound
import time

def set_gpu(gpus):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus


def make_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def notify(times=5):
    for i in range(times):
        time.sleep(0.5)
        winsound.Beep(1000, 1000)  # 1000Hz, 持续1秒
        time.sleep(0.5)
        winsound.Beep(800, 500)  # 稍高音调，但时间短


def safe_empty_cache():
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"Cache cleanup failed: {e}")
        torch.cuda.set_device(torch.cuda.current_device())


def set_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def seed_worker(worker_id):
    """DataLoader worker的种子函数"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)