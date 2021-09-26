import os
import os.path
import numpy as np
import joblib
from pathlib import Path
import tqdm
import torch
from torch.autograd import Variable

#%%

def get(batch_size, data_root, train=False, val=True, **kwargs):
    print("Building IMAGENET data loader, 50000 for test")
    ds = []
    assert train is not True, 'train not supported yet'
    if train:
        ds.append(IMAGENET(data_root, batch_size, True, **kwargs))
    if val:
        ds.append(IMAGENET(data_root, batch_size, False, **kwargs))
    ds = ds[0] if len(ds) == 1 else ds
    return ds

class IMAGENET(object):
    def __init__(self, root, batch_size, train=False, input_size=224, **kwargs):
        self.mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)
        self.train = train

        if train:
            pkl_file = os.path.join(root, 'train{}.pkl'.format(input_size))
        else:
            pkl_file = os.path.join(root, 'val{}.pkl'.format(input_size))
        self.data_dict = joblib.load(pkl_file)

        self.batch_size = batch_size
        self.idx = 0

    @property
    def n_batch(self):
        return int(np.ceil(self.n_sample* 1.0 / self.batch_size))

    @property
    def n_sample(self):
        return len(self.data_dict['data'])

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= self.n_batch:
            self.idx = 0
            raise StopIteration
        else:
            img = self.data_dict['data'][self.idx*self.batch_size:(self.idx+1)*self.batch_size].astype('float32')
            target = self.data_dict['target'][self.idx*self.batch_size:(self.idx+1)*self.batch_size]
            self.idx += 1
            return img, target

if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parent.parent
    data_root = project_dir / 'data' / 'imagenet-val'
    val_ds = get(100, data_root)
    print('n_sample', val_ds.n_sample)
    print('n_batch', val_ds.n_batch)
    
    n_passed = 0
    n_sample = len(val_ds)
    for idx, (data, target) in enumerate(tqdm.tqdm(val_ds, total=n_sample)):
        n_passed += len(data)
        data =  Variable(torch.FloatTensor(data)).cuda()
        indx_target = torch.LongTensor(target)
        
        if idx >= n_sample - 1:
            break