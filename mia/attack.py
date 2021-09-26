import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import TensorDataset
from sklearn.metrics import classification_report

import model
from model import init_params as w_init
from train import  train_attack_model, prepare_attack_data

import sys 
sys.path.append("..") 
from utils.dataloader_util import generate
from utils.train_util import load_model_pytorch
from nets.resnet_cifar import ResNet_CIFAR
from nets.vgg import VGG_CIFAR

#%%

#set the seed for reproducibility
np.random.seed(1234)
#Flag to enable early stopping
need_earlystop = False

################################
#Attack Model Hyperparameters
################################
#Number of processes
num_workers = 2
NUM_EPOCHS = 2
BATCH_SIZE = 10
#Learning rate
LR_ATTACK = 0.001 
#L2 Regulariser
REG = 1e-7
#weight decay
LR_DECAY = 0.96
#No of hidden units
n_hidden = 128
#Binary Classsifier
out_classes = 2


def get_arguments():
    parser = argparse.ArgumentParser(prog="Membership Inference Attack")
    parser.add_argument('--dataset', type=str, default='cifar10', help='Which dataset to use (cifar10 or imagenet)')
    parser.add_argument('--dataPath', type=str, help='Path to store data')
    parser.add_argument('--model', type=str, default='resnet56', help='model to use (resnet56 or resnet20)')
    parser.add_argument('--modelPath', type=str, help='Path to load model checkpoints')
    parser.add_argument('--ckptPath', type=str, help='Path to save attack model checkpoints')
    parser.add_argument('--batch_size', type=int, default=128, help='trainset batch size (default 128)')
    parser.add_argument('--test_batch_size', type=int, default=256, help='testset batch size (default 256)')

    parser.add_argument('--seed', type=int, default=0, help='random seed (default 0)')
    parser.add_argument('--unlearn_class', type=int, help='class label to unlearn')
    
    parser.add_argument('--need_topk',action='store_true', help='Flag to enable using Top 3 posteriors for attack data')
    parser.add_argument('--param_init', action='store_true', help='Flag to enable custom model params initialization')
    parser.add_argument('--verbose',action='store_true', help='Add Verbosity')
    return parser.parse_args()    


def attack_inference(model,
                    test_X,
                    test_Y,
                    device):
    
    print('----Attack Model Testing----')

    targetnames= ['Non-Member', 'Member']
    pred_y = []
    true_y = []
    
    #Tuple of tensors
    X = torch.cat(test_X)
    Y = torch.cat(test_Y)
    
    #Create Inference dataset
    inferdataset = TensorDataset(X,Y) 
    dataloader = torch.utils.data.DataLoader(dataset=inferdataset,
                                            batch_size=50,
                                            shuffle=False,
                                            num_workers=num_workers)

    #Evaluation of Attack Model
    model.eval() 
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            
            #Predictions for accuracy calculations
            _, predictions = torch.max(outputs.data, 1)
            total+=labels.size(0)
            correct+=(predictions == labels).sum().item()
            
            #print('True Labels for Batch [{}] are : {}'.format(i,labels))
            #print('Predictions for Batch [{}] are : {}'.format(i,predictions))
            
            true_y.append(labels.cpu())
            pred_y.append(predictions.cpu())
        
    attack_acc = correct / total
    print('Attack Test Accuracy is  : {:.2f}%'.format(100*attack_acc))
    
    true_y =  torch.cat(true_y).numpy()
    pred_y = torch.cat(pred_y).numpy()

    #print('---Detailed Results----')
    #print(classification_report(true_y, pred_y, target_names=targetnames))


#Main Method to initate model and attack
def create_attack(args):
 
    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    
    #Load the saved checkpoint of target model
    print('Use Target model at the path ====> [{}] '.format(args.modelPath))
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
        total_classes = 10 # [0-9]
        trainset = torchvision.datasets.CIFAR10(root=args.dataPath, train=True, download=False, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=args.dataPath, train=False, download=False, transform=transform_test)

        if args.model == 'resnet56':
            target_model = ResNet_CIFAR(depth=56, num_classes=10)
        elif args.model == 'resnet20':
            target_model = ResNet_CIFAR(depth=20, num_classes=10)
        elif args.model == 'vgg':
            target_model = VGG_CIFAR(num_classes=10)
        else:
            print('no model')
    
    target_model = target_model.cuda()
    load_model_pytorch(target_model, args.modelPath, args.model)  
    
    list_allclasses = list(range(total_classes))
    unlearn_listclass = [args.unlearn_class]
    list_allclasses.remove(args.unlearn_class) # rest classes
    rest_traindata = generate(trainset, list_allclasses)
    rest_testdata = generate(testset, list_allclasses)
    unlearn_traindata = generate(trainset, unlearn_listclass)
    unlearn_testdata = generate(testset, unlearn_listclass)
    
    print('Total Train samples in {} dataset: {}'.format(args.dataset, len(trainset))) 
    print('Total Test samples in {} dataset: {}'.format(args.dataset, len(testset))) 
    print('Number of unlearn train samples: {}'.format(len(unlearn_traindata)))
    print('Number of unlearn test samples: {}'.format(len(unlearn_testdata)))
    print('Number of rest train samples: {}'.format(len(rest_traindata)))
    print('Number of rest test samples: {}'.format(len(rest_testdata)))
    
    rest_trainloader = torch.utils.data.DataLoader(rest_traindata, batch_size=args.batch_size, 
                                                   shuffle=False, num_workers=4)
    rest_testloader = torch.utils.data.DataLoader(rest_testdata, batch_size=args.test_batch_size, 
                                                  shuffle=False, num_workers=4)
    unlearn_trainloader = torch.utils.data.DataLoader(unlearn_traindata, batch_size=args.batch_size, 
                                                  shuffle=False, num_workers=4)
    unlearn_testloader = torch.utils.data.DataLoader(unlearn_testdata, batch_size=args.test_batch_size, 
                                                  shuffle=False, num_workers=4)
    
    print('---Preparing Attack Training data---')
    rest_trainX, rest_trainY = prepare_attack_data(target_model, rest_trainloader, device, args.need_topk)
    rest_testX, rest_testY = prepare_attack_data(target_model, rest_testloader, device, args.need_topk, unused_dataset=True)
    attack_trainX = rest_trainX + rest_testX
    attack_trainY = rest_trainY + rest_testY
    
    print('---Preparing Attack Testing data---')
    unlearn_trainX, unlearn_trainY = prepare_attack_data(target_model, unlearn_trainloader, device, args.need_topk, unused_dataset=True)
    unlearn_testX, unlearn_testY = prepare_attack_data(target_model, unlearn_testloader, device, args.need_topk, unused_dataset=True)
    attack_testX = unlearn_trainX + unlearn_testX
    attack_testY = unlearn_trainY + unlearn_testY
    
    
    ###################################
    # Attack Model Training
    ##################################
    #The input dimension to MLP attack model
    input_size = attack_trainX[0].size(1)
    print('Input Feature dim for Attack Model : [{}]'.format(input_size))
    
    attack_model = model.AttackMLP(input_size,n_hidden,out_classes).to(device)
    
    if (args.param_init):
        #Initialize params
        attack_model.apply(w_init)

    # Loss and optimizer
    attack_loss = nn.CrossEntropyLoss()
    attack_optimizer = torch.optim.Adam(attack_model.parameters(), lr=LR_ATTACK, weight_decay=REG)
    attack_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(attack_optimizer,gamma=LR_DECAY)

    #Feature vector and labels for training Attack model
    attackdataset = (attack_trainX, attack_trainY)
    
    attack_valacc = train_attack_model(attack_model, attackdataset, attack_loss,
                       attack_optimizer, attack_lr_scheduler, device, args.ckptPath,
                        NUM_EPOCHS, BATCH_SIZE, num_workers, args.verbose)
  
    print('Validation Accuracy for the Best Attack Model is: {:.2f} %'.format(100* attack_valacc))
   
    #Load the trained attack model
    attack_model.load_state_dict(torch.load(args.ckptPath))
    
    #Inference on trained attack model
    attack_inference(attack_model, attack_testX, attack_testY, device)


if __name__ == '__main__':
    args = get_arguments()
    project_dir = Path(__file__).resolve().parent.parent
    args.dataPath = project_dir / 'data'
    args.model = 'resnet20'
    #args.modelPath = project_dir / 'ckpt' / args.model / 'seed_0_acc_84.15.pth'
    args.modelPath = project_dir / 'ckpt' / 'finetuned'/ args.model / 'seed0_acc86.68_epoch9_2021-09-23 18-47-59.pth'
    args.ckptPath = project_dir / 'mia' / 'model_ckpt' / 'best_attack_model.ckpt'
    args.seed = 1234
    args.unlearn_class = 9
    print(args)
    np.random.seed(args.seed)
    
    #Generate Membership inference attack1
    create_attack(args)
    