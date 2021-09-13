import os
import random
import argparse
from pathlib import Path
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.backends import cudnn
from utils.train_util import load_model_pytorch, train, test
from nets.resnet_cifar import ResNet_CIFAR
from nets.vgg import VGG_CIFAR
from cdp import acculumate_feature, calculate_cdp, \
    get_threshold_by_sparsity, TFIDFPruner

#%%
'''
    Configuration
'''
parser = argparse.ArgumentParser(description='Class Pruning')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='cifar10 or imagenet')
parser.add_argument('--dataroot', type=str, metavar='PATH',
                    help='Path to Dataset folder')
parser.add_argument('--model', type=str, default='resnet56',
                    help='model to use, only resnet56, resnet20')
parser.add_argument('--pretrained_dir', type=str, default=None,
                    help='pretrained file path')
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for statistics (default: 128)')
parser.add_argument('--stop_batch', type=int, default=200, 
                    help="Sample batch number (default: 200)")
parser.add_argument('--search_batch_size', type=int, default=256,
                    help='input batch size for search (default: 256)')
parser.add_argument('--test_batch_size', type=int, default=256,
                    help='input batch size for testing (default: 256)')
parser.add_argument('--gpus', default=None, 
                    help='List of GPUs used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='Number of data loading workers (default: 8)')
parser.add_argument('--seed', type=int, default=0, 
                    help='random seed (default: 0)')
parser.add_argument('--epochs', type=int, default=300,
                    help='epochs to fine tune (default: 300)')
parser.add_argument('--unlearn_class', type=int,
                    help='class label to unlearn')
parser.add_argument('--coe', type=int,
                    help='whether to use balance coefficient')
parser.add_argument('--sparsity', type=float, default=0.39,
                    help='target overall target sparsity')
parser.add_argument('--save_acc', type=float, default=94.0,
                    help='save accuracy')
parser.add_argument('--savepath', type=str, default='./ckpt/',
                    help='model save directory')

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     torch.cuda.manual_seed(seed)
     np.random.seed(seed)
     random.seed(seed)
     cudnn.deterministic = True

def load_model_CIFAR10(args):
    if args.model == 'resnet56':
        net = ResNet_CIFAR(depth=56, num_classes=10)
        model_path = './models/resnet56_base/checkpoint/model_best.pth.tar'
    elif args.model == 'resnet20':
        net = ResNet_CIFAR(depth=20, num_classes=10)
        model_path = './models/resnet20_base/checkpoint/model_best.pth.tar'
    elif args.model == 'vgg':
        net = VGG_CIFAR(num_classes=10)
        model_path = './models/vgg_base/checkpoint/model_best.pth.tar'
    else:
        print('no model')
        return
    if args.pretrained_dir:
        model_path = args.pretrained_dir
    net = net.cuda()
    load_model_pytorch(net, model_path, args.model)
    return net

def Class_Pruning():
    '''configuration'''
    args = parser.parse_args()
    args.dataset = 'cifar10'
    project_dir = Path(__file__).resolve().parent
    args.dataroot = project_dir / 'data'
    args.model = 'resnet20'
    args.pretrained_dir = project_dir / 'ckpt' / 'base' / 'resnet20_model_base.th'
    args.gpus = 0
    args.j = 4
    args.stop_batch = 1
    args.unlearn_class = 9
    args.sparsity = 0.5
    #args.coe = 0
    print(args)
    setup_seed(args.seed)
    save_file = '_'.join([str(args.model),
                      'coe{}'.format(args.coe),
                      'seed{}'.format(args.seed)
                      ])
    args.savepath=os.path.join(args.savepath,args.model)
    save_info = os.path.join(args.savepath, save_file)
    save_acc = args.save_acc   
    
    if args.dataset == 'cifar10':
        '''load data and model'''
        mean=[125.31 / 255, 122.95 / 255, 113.87 / 255]
        std=[63.0 / 255, 62.09 / 255, 66.70 / 255]
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
            ])
        trainset = torchvision.datasets.CIFAR10(root=args.dataroot, train=True, download=False, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=args.dataroot, train=False, download=False, transform=transform_test)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        train_all_loader = torch.utils.data.DataLoader(trainset, batch_size=args.search_batch_size, shuffle=False, num_workers=4)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=4)
        net = load_model_CIFAR10(args)
        
        '''pre-processing'''
        feature_iit, classes = acculumate_feature(net, train_all_loader, args.stop_batch)
        tf_idf_map = calculate_cdp(feature_iit, classes, args.dataset, args.coe, unlearn_class=args.unlearn_class)
        #threshold = get_threshold_by_sparsity(tf_idf_map, args.sparsity)
    

    #flops, param, detail_flops = count_flops_params(net, (1, 3, 32, 32))
    
    
    
    print('fuck')

if __name__=='__main__':
    Class_Pruning()