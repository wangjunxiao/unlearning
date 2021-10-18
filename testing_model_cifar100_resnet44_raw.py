import random
import argparse
from pathlib import Path
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.backends import cudnn
from utils.dataloader_util import generate
from utils.train_util import load_model_pytorch, test
from nets.resnet_cifar import ResNet_CIFAR
from nets.vgg import VGG_CIFAR

#%%
'''
    Configuration
'''
parser = argparse.ArgumentParser(description='Training model')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='cifar10 or imagenet')
parser.add_argument('--dataroot', type=str, metavar='PATH',
                    help='Path to Dataset folder')
parser.add_argument('--model', type=str, default='resnet56',
                    help='model to use, only resnet56, resnet20')
parser.add_argument('--test_batch_size', type=int, default=256,
                    help='input batch size for testing (default: 256)')
parser.add_argument('--gpus', default=None, 
                    help='List of GPUs used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='Number of data loading workers (default: 8)')
parser.add_argument('--seed', type=int, default=0, 
                    help='random seed (default: 0)')
parser.add_argument('--unlearn_class', type=int,
                    help='class label to unlearn')


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     torch.cuda.manual_seed(seed)
     np.random.seed(seed)
     random.seed(seed)
     cudnn.deterministic = True

def Testing():
    '''configuration'''
    args = parser.parse_args()
    args.dataset = 'cifar100'
    project_dir = Path(__file__).resolve().parent
    args.dataroot = project_dir / 'data'
    args.model = 'resnet44'
    args.gpus = 0
    args.unlearn_class = 99
    args.j = 4
    print(args)

    setup_seed(args.seed)
    model_path = project_dir / 'ckpt' / args.model / 'cifar100' / 'seed0_acc50.83_epoch27_2021-10-18 12-22-27.pth' 
    
    if args.dataset == 'cifar10':
        '''load data and model'''
        mean=[125.31 / 255, 122.95 / 255, 113.87 / 255]
        std=[63.0 / 255, 62.09 / 255, 66.70 / 255]
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
            ])
        total_classes = 10 # [0-9]
        testset = torchvision.datasets.CIFAR10(root=args.dataroot, train=False, download=False, transform=transform_test)

        if args.model == 'resnet56':
            net = ResNet_CIFAR(depth=56, num_classes=10)
        elif args.model == 'resnet20':
            net = ResNet_CIFAR(depth=20, num_classes=10)
        elif args.model == 'resnet32':
            net = ResNet_CIFAR(depth=32, num_classes=10)
        elif args.model == 'vgg':
            net = VGG_CIFAR(num_classes=10)
        else:
            print('no model')
            
    if args.dataset == 'cifar100':
        '''load data and model'''
        mean=[129.3 / 255, 124.1 / 255, 112.4 / 255]
        std=[68.2 / 255, 65.4 / 255, 70.4 / 255]
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
            ])
        total_classes = 100 # [0-99]
        testset = torchvision.datasets.CIFAR100(root=args.dataroot, train=False, download=False, transform=transform_test)
        
        if args.model == 'resnet56':
            net = ResNet_CIFAR(depth=56, num_classes=100)
        elif args.model == 'resnet20':
            net = ResNet_CIFAR(depth=20, num_classes=100)
        elif args.model == 'resnet32':
            net = ResNet_CIFAR(depth=32, num_classes=100)
        elif args.model == 'resnet44':
            net = ResNet_CIFAR(depth=44, num_classes=100)
        elif args.model == 'vgg':
            net = VGG_CIFAR(num_classes=100)
        else:
            print('no model')    
            
    net = net.cuda()
    load_model_pytorch(net, model_path, args.model)

    list_allclasses = list(range(total_classes))
    unlearn_listclass = [args.unlearn_class]
    list_allclasses.remove(args.unlearn_class) # rest classes
    rest_testdata = generate(testset, list_allclasses)
    unlearn_testdata = generate(testset, unlearn_listclass)
    print(len(rest_testdata), len(unlearn_testdata))
    rest_testloader = torch.utils.data.DataLoader(rest_testdata, batch_size=args.test_batch_size, 
                                                  shuffle=False, num_workers=4)
    unlearn_testloader = torch.utils.data.DataLoader(unlearn_testdata, batch_size=args.test_batch_size, 
                                                  shuffle=False, num_workers=4)

    print('*'*5+'testing in unlearn_data'+'*'*12)
    test(net, unlearn_testloader)
    print('*'*40)
    
    print('*'*5+'testing in rest_data'+'*'*12)
    test(net, rest_testloader)
    print('*'*40)

if __name__=='__main__':
    Testing()

