import torch.nn as nn

#%%

#Below methods to claculate input featurs to the FC layer
#and weight initialization for CNN model is based on the below github repo
#Based on :https://github.com/Lab41/cyphercat/blob/master/Utils/models.py
 
def size_conv(size, kernel, stride=1, padding=0):
    out = int(((size - kernel + 2*padding)/stride) + 1)
    return out
    
    
def size_max_pool(size, kernel, stride=None, padding=0): 
    if stride == None: 
        stride = kernel
    out = int(((size - kernel + 2*padding)/stride) + 1)
    return out

#Calculate in_features for FC layer in Shadow Net
def calc_feat_linear_cifar(size):
    feat = size_conv(size,3,1,1)
    feat = size_max_pool(feat,2,2)
    feat = size_conv(feat,3,1,1)
    out = size_max_pool(feat,2,2)
    return out
    
#Calculate in_features for FC layer in Shadow Net
def calc_feat_linear_mnist(size):
    feat = size_conv(size,5,1)
    feat = size_max_pool(feat,2,2)
    feat = size_conv(feat,5,1)
    out = size_max_pool(feat,2,2)
    return out

#Parameter Initialization
def init_params(m): 
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear): 
        nn.init.xavier_normal_(m.weight.data)
        nn.init.zeros_(m.bias)

#####################################################
# Define Attack Model Architecture
#####################################################

#Attack MLP Model
class AttackMLP(nn.Module):
    def __init__(self, input_size, hidden_size=64,out_classes=2):
        super(AttackMLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, out_classes)
        )    
    def forward(self, x):
        out = self.classifier(x)
        return out          
    
