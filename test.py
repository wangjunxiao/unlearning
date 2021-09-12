import torch
import numpy as np
import torch.nn.functional as F

a = np.array([[5, 5, 5], [5, 5, 5]])
b = np.array([5, 5, 5])
aa = torch.from_numpy(a)
bb = torch.from_numpy(b)

f = F.relu(aa)
print(f.view(2,-1))

