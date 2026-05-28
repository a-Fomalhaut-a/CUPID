from torch.utils.data import DataLoader
import numpy as np

try:
    from config_train import train_opt
except ImportError:
    from config import train_opt

def datacreate(dataset, flag='train', batch_size=None):
    if flag == 'train':
        if batch_size is None:
            batch_size = train_opt['batchsize']
        datatr = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True, pin_memory=True, worker_init_fn=lambda _: np.random.seed(42))
    elif flag == 'validation':
        datatr = DataLoader(dataset['validation'], batch_size=1, shuffle=False, pin_memory=True)
    else:
        datatr = DataLoader(dataset['test'], batch_size=1, shuffle=False, pin_memory=True)
    return datatr

