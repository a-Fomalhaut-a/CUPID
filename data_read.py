from torch.utils.data import DataLoader
import numpy as np
from config import train_opt

def datacreate(dataset,flag='train'):
    if flag == 'train':
        datatr = DataLoader(dataset['train'], batch_size=train_opt['batchsize'], shuffle=True, pin_memory=True, worker_init_fn=lambda _: np.random.seed(42))
    elif flag == 'validation':
        datatr = DataLoader(dataset['validation'], batch_size=1, shuffle=False, pin_memory=True)
    else:
        datatr = DataLoader(dataset['test'], batch_size=1, shuffle=False, pin_memory=True)
    return datatr

