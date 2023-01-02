
import torch
import random
import os
import numpy as np


def get_logger(filename=OUTPUT_DIR+'train'):
    '''
    :用来打印和记录一些日志
    :用法如下
        LOGGER = get_logger()
        LOGGER.info(f'Score: {score:<.4f}')
    '''
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def seed_everything(seed=42):
    '''
    :设定随机种子
    ：用法如下
        seed_everything(seed=42)
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 环境变量
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class AverageMeter(object):
    """Computes and stores the average and current value 
    ：累加器 可用于计算loss 精度等
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val  # 当前的值-by每个样本
        self.sum += val * n  # 值加和
        self.count += n  # 样本数量加和
        self.avg = self.sum / self.count  # 计算到此的均值

