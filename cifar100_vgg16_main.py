import time
import random
import argparse
from pathlib import Path
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.backends import cudnn
from utils.train_util import load_model_pytorch, train, test
from utils.dataloader_util import generate
from nets.resnet_cifar import ResNet_CIFAR
from nets.vgg import VGG_CIFAR
from class_pruner import acculumate_feature, calculate_cp, \
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
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate to fine tune (default: 0.1)')
parser.add_argument('--unlearn_class', type=int,
                    help='class label to unlearn')
parser.add_argument('--coe', type=int,
                    help='whether to use balance coefficient')
parser.add_argument('--sparsity', type=float, default=0.39,
                    help='target overall target sparsity')
parser.add_argument('--save_acc', type=float, default=94.0,
                    help='save accuracy')
parser.add_argument('--label_smoothing', type=float, default='0.0',
                    help='label smoothing rate')
parser.add_argument('--warmup_step', type=int, default='0',
                    help='warm up epochs')
parser.add_argument('--warm_lr', type=float, default='10e-5',
                    help='warm up learning rate')
parser.add_argument('--model_file', type=str,
                    help='model file name')

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     torch.cuda.manual_seed(seed)
     np.random.seed(seed)
     random.seed(seed)
     cudnn.deterministic = True

def load_model_CIFAR10(args, model_path):
    if args.model == 'resnet56':
        net = ResNet_CIFAR(depth=56, num_classes=10)
    elif args.model == 'resnet20':
        net = ResNet_CIFAR(depth=20, num_classes=10)
    elif args.model == 'vgg':
        net = VGG_CIFAR(num_classes=10)
    else:
        print('no model')
        return
    net = net.cuda()
    load_model_pytorch(net, model_path, args.model)
    return net

def load_model_CIFAR100(args, model_path):
    if args.model == 'resnet56':
        net = ResNet_CIFAR(depth=56, num_classes=100)
    elif args.model == 'resnet20':
        net = ResNet_CIFAR(depth=20, num_classes=100)
    elif args.model == 'vgg16':
        net = VGG_CIFAR(cfg_index=16, num_classes=100)
    else:
        print('no model')
        return
    net = net.cuda()
    load_model_pytorch(net, model_path, args.model)
    return net

def Class_Pruning():
    '''configuration'''
    args = parser.parse_args()
    args.dataset = 'cifar100'
    project_dir = Path(__file__).resolve().parent
    args.dataroot = project_dir / 'data'
    args.model = 'vgg16'
    args.gpus = 0
    args.j = 4
    args.stop_batch = 20
    args.unlearn_class = 99
    args.sparsity = 0.13   #cifar10 args.sparsity = 0.05
    args.coe = 0
    args.epochs = 10
    args.lr = 0.1
    args.save_acc = 0.0
    args.label_smoothing = 0.0 
    args.warmup_step = 0
    args.warm_lr = 10e-5
    args.model_file = 'seed0_acc50.28_epoch18_2021-10-14 16-18-36.pth'
    print(args)
    
    setup_seed(args.seed)
    model_path = project_dir / 'ckpt' / args.model / 'cifar100' / args.model_file 
    pruned_save_info = project_dir / 'ckpt' / 'pruned' / args.model / 'cifar100'
    finetuned_save_info = project_dir / 'ckpt' / 'finetuned' / args.model / 'cifar100'   
    
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
        net = load_model_CIFAR10(args, model_path)
        total_classes = 10 # [0-9]
    
    if args.dataset == 'cifar100':
        '''load data and model'''
        mean=[129.3 / 255, 124.1 / 255, 112.4 / 255]
        std=[68.2 / 255, 65.4 / 255, 70.4 / 255]
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
        trainset = torchvision.datasets.CIFAR100(root=args.dataroot, train=True, download=False, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=args.dataroot, train=False, download=False, transform=transform_test)
        net = load_model_CIFAR100(args, model_path)
        total_classes = 100 # [0-99]
        
    #trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    train_all_loader = torch.utils.data.DataLoader(trainset, batch_size=args.search_batch_size, shuffle=False, num_workers=4)
    #testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=4)
    
    '''pre-processing'''
    feature_iit, classes = acculumate_feature(net, train_all_loader, args.stop_batch)
    tf_idf_map = calculate_cp(feature_iit, classes, args.dataset, args.coe, unlearn_class=args.unlearn_class)
    threshold = get_threshold_by_sparsity(tf_idf_map, args.sparsity)
    print('threshold', threshold)

    '''test before pruning'''
    list_allclasses = list(range(total_classes))
    unlearn_listclass = [args.unlearn_class]
    list_allclasses.remove(args.unlearn_class) # rest classes
    unlearn_testdata = generate(testset, unlearn_listclass)
    rest_testdata = generate(testset, list_allclasses)
    print(len(unlearn_testdata), len(rest_testdata))
    unlearn_testloader = torch.utils.data.DataLoader(unlearn_testdata, batch_size=args.test_batch_size, 
                                                     shuffle=False, num_workers=4)
    rest_testloader = torch.utils.data.DataLoader(rest_testdata, batch_size=args.test_batch_size, 
                                                     shuffle=False, num_workers=4)
    print('*'*5+'testing in unlearn_data'+'*'*12)
    test(net, unlearn_testloader)
    print('*'*40)
    print('*'*5+'testing in rest_data'+'*'*15)
    test(net, rest_testloader)
    print('*'*40)
    
    '''pruning''' 
    cp_config={ "threshold": threshold, "map": tf_idf_map }
    config_list = [{
        'sparsity': args.sparsity,  
        'op_types': ['Conv2d']
        }]
    pruner = TFIDFPruner(net, config_list, cp_config=cp_config)
    pruner.compress()
    pruned_model_path = pruned_save_info / ('seed_'+str(args.seed)+
                                            time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())+
                                            '_model.pth')
    pruned_mask_path = pruned_save_info / ('seed_'+str(args.seed)+
                                           time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())+
                                            '_mask.pth')
    pruner.export_model(pruned_model_path, pruned_mask_path)
    pruned_net = load_model_CIFAR100(args, pruned_model_path)
    
    '''test after pruning'''
    print('*'*5+'testing in unlearn_data'+'*'*12)
    test(pruned_net, unlearn_testloader)
    print('*'*40)
    print('*'*5+'testing in rest_data'+'*'*15)
    test(pruned_net, rest_testloader)
    print('*'*40)
    
    #return #for test
    
    '''fine tuning'''
    rest_traindata = generate(trainset, list_allclasses)
    print(len(rest_traindata))
    rest_trainloader = torch.utils.data.DataLoader(rest_traindata, batch_size=args.batch_size, 
                                                     shuffle=False, num_workers=4)
    train(pruned_net, epochs=args.epochs, lr=args.lr, train_loader=rest_trainloader, 
          test_loader=rest_testloader, save_info=finetuned_save_info, save_acc=args.save_acc, seed=args.seed,
          label_smoothing=args.label_smoothing, warmup_step=args.warmup_step, warm_lr=args.warm_lr)
    
    '''test after fine-tuning'''
    print('*'*5+'testing in unlearn_data'+'*'*12)
    test(pruned_net, unlearn_testloader)
    print('*'*40)
    print('*'*5+'testing in rest_data'+'*'*15)
    test(pruned_net, rest_testloader)
    print('*'*40)
    
    print('finished')
    
if __name__=='__main__':
    Class_Pruning()
    