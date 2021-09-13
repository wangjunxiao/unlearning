import torch
import numpy as np
import torch.nn.functional as F

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
print(str(save_acc)[0:5])