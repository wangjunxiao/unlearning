# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#%%

"""Function: load data"""
def data_init(FL_params):
    kwargs = {'num_workers': 0, 'pin_memory': True} if FL_params.cuda_state else {}
    trainset, testset = data_set(FL_params.data_name)

    #Testset dataloader
    test_loader = DataLoader(testset, batch_size=FL_params.test_batch_size, shuffle=True, **kwargs)
    
    #Evenly distribute trainset into clients
    split_index = [int(trainset.__len__()/FL_params.N_total_client)]*(FL_params.N_total_client-1)
    split_index.append(int(trainset.__len__() - int(trainset.__len__()/FL_params.N_total_client)*(FL_params.N_total_client-1)))
    client_dataset = torch.utils.data.random_split(trainset, split_index)
   
    client_loaders = []
    for i in range(FL_params.N_total_client):
        client_loaders.append(DataLoader(client_dataset[i], FL_params.local_batch_size, shuffle=True, **kwargs))
        '''
        By now，the private data of clients is well partitioned and stored in client_loaders. 
        Each element corresponds to the private data of a certain client.
        '''
    
    return client_loaders, test_loader


def data_set(data_name):
    if not data_name in ['mnist', 'cifar10']:
        raise TypeError('data_name should be a string, including mnist and cifar10.')
    
    #model: 2 conv. layers followed by 2 FC layers
    if(data_name == 'mnist'):
        transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))])
        trainset = datasets.MNIST(root='~\.torch', train=True, 
                                                   download=True, transform=transform)

        testset  = datasets.MNIST(root='~\.torch', train=False, 
                                                   download=True, transform=transform)

    #model: 2 conv. layers followed by 3 FC layers
    elif(data_name == 'cifar10'):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = datasets.CIFAR10(root='~\.torch', train=True,
                                                     download=True, transform=transform)
        
        testset  = datasets.CIFAR10(root='~\.torch', train=False,
                                                     download=True, transform=transform)
        
    return trainset, testset


