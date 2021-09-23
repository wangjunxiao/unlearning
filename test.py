import time
import torch
import numpy as np
import torch.nn.functional as F

import importlib
import logging
importlib.reload(logging)
logpath = ('./log/'+time.strftime("%m-%d %Hh%Mm%Ss", time.localtime())+'.log').replace('\\','/')
logging.basicConfig(
    filename=logpath,
    filemode='a',
    format='%(message)s',
    level=logging.INFO,
    datefmt='%H%M%S')

a = np.array([[5., 5., 5.], [5., 5., 5.]])
b = np.array([5, 5, 5])
aa = torch.from_numpy(a)
bb = torch.from_numpy(b)

aa[1] = aa[1]/2

print (aa)

print(torch.zeros(2, 3))

#f = F.relu(aa)
#print(f.view(2,-1))

save_acc = 34.53434
for i in range(10):
    #time.sleep(1)
    logging.info('fuck')
print(str(save_acc)[0:5])

coe = 0
print(coe if coe else 1.0)