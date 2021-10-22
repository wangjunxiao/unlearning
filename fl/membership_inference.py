# -*- coding: utf-8 -*-
import torch
from torch.nn.functional import softmax
import numpy as np

from sklearn.metrics import precision_score, recall_score
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

#%%

def attack(target_model, attack_model, client_loaders, test_loader, FL_params):
    n_class_dict = dict()
    n_class_dict['adult'] = 2
    n_class_dict['purchase'] = 2
    n_class_dict['mnist'] = 10
    n_class_dict['cifar10'] = 10
    
    N_class = n_class_dict[FL_params.data_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    target_model.to(device)
        
    target_model.eval()
    
    #The predictive output of forgotten user data after passing through the target model.
    unlearn_X = torch.zeros([1,N_class])
    unlearn_X = unlearn_X.to(device)
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(client_loaders[FL_params.forget_client_idx]):
                    data = data.to(device)
                    out = target_model(data)
                    unlearn_X = torch.cat([unlearn_X, out])
                    
    unlearn_X = unlearn_X[1:,:]
    unlearn_X = softmax(unlearn_X,dim = 1)
    unlearn_X = unlearn_X.cpu().detach().numpy()
    
    unlearn_X.sort(axis=1)
    unlearn_y = np.ones(unlearn_X.shape[0])
    unlearn_y = unlearn_y.astype(np.int16)
    
    N_unlearn_sample = len(unlearn_y)
    
    #Test data, predictive output obtained after passing the target model
    test_X = torch.zeros([1, N_class])
    test_X = test_X.to(device)
    with torch.no_grad():
        for _, (data, target) in enumerate(test_loader):
            data = data.to(device)
            out = target_model(data)
            test_X = torch.cat([test_X, out])
            
            if(test_X.shape[0] > N_unlearn_sample):
                break
    test_X = test_X[1:N_unlearn_sample+1,:]
    test_X = softmax(test_X,dim = 1)
    test_X = test_X.cpu().detach().numpy()
    
    test_X.sort(axis=1)
    test_y = np.zeros(test_X.shape[0])
    test_y = test_y.astype(np.int16)
    
    #The data of the forgotten user passed through the output of the target model, and the data of the test set passed through the output of the target model were spliced together
    #The balanced data set that forms the 50% train 50% test.
    XX = np.vstack((unlearn_X, test_X))
    YY = np.hstack((unlearn_y, test_y))
    
    pred_YY = attack_model.predict(XX)
    # acc = accuracy_score( YY, pred_YY)
    pre = precision_score(YY, pred_YY, pos_label=1)
    rec = recall_score(YY, pred_YY, pos_label=1)
    # print("MIA Attacker accuracy = {:.4f}".format(acc))
    print("MIA Attacker precision = {:.4f}".format(pre))
    print("MIA Attacker recall = {:.4f}".format(rec))
    
    return (pre, rec)


def train_attack_model(shadow_old_GM, shadow_client_loaders, shadow_test_loader, FL_params):
    shadow_model = shadow_old_GM
    n_class_dict = dict()
    n_class_dict['adult'] = 2
    n_class_dict['purchase'] = 2
    n_class_dict['mnist'] = 10
    n_class_dict['cifar10'] = 10
    
    N_class = n_class_dict[FL_params.data_name]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shadow_model.to(device)
        
    shadow_model.eval()
    ####
    pred_4_mem = torch.zeros([1,N_class])
    pred_4_mem = pred_4_mem.to(device)
    with torch.no_grad():
        for ii in range(len(shadow_client_loaders)):
            # if(ii != FL_params.forget_client_idx):
            #     continue
            data_loader = shadow_client_loaders[ii]
            
            for batch_idx, (data, target) in enumerate(data_loader):
                    data = data.to(device)
                    out = shadow_model(data)
                    pred_4_mem = torch.cat([pred_4_mem, out])
    pred_4_mem = pred_4_mem[1:,:]
    pred_4_mem = softmax(pred_4_mem,dim = 1)
    pred_4_mem = pred_4_mem.cpu()
    pred_4_mem = pred_4_mem.detach().numpy()
    
    ####
    pred_4_nonmem = torch.zeros([1,N_class])
    pred_4_nonmem = pred_4_nonmem.to(device)
    with torch.no_grad():
        for batch, (data, target) in enumerate(shadow_test_loader):
            data = data.to(device)
            out = shadow_model(data)
            pred_4_nonmem = torch.cat([pred_4_nonmem, out])
    pred_4_nonmem = pred_4_nonmem[1:,:]
    pred_4_nonmem = softmax(pred_4_nonmem,dim = 1)
    pred_4_nonmem = pred_4_nonmem.cpu()
    pred_4_nonmem = pred_4_nonmem.detach().numpy()
    
    
    #build MIA attack model 
    att_y = np.hstack((np.ones(pred_4_mem.shape[0]), np.zeros(pred_4_nonmem.shape[0])))
    att_y = att_y.astype(np.int16)
    
    att_X = np.vstack((pred_4_mem, pred_4_nonmem))
    att_X.sort(axis=1)
    
    X_train,X_test, y_train, y_test = train_test_split(att_X, att_y, test_size = 0.1)
    
    attacker = XGBClassifier(n_estimators = 300,
                              n_jobs = -1,
                                max_depth = 30,
                              objective = 'binary:logistic',
                              booster="gbtree",
                              # learning_rate=None,
                               # tree_method = 'gpu_hist',
                               scale_pos_weight = pred_4_nonmem.shape[0]/pred_4_mem.shape[0]
                              )
    
    attacker.fit(X_train, y_train)
    # print('\n')
    # print("MIA Attacker training accuracy")
    # print(accuracy_score(y_train, attacker.predict(X_train)))
    # print("MIA Attacker testing accuracy")
    # print(accuracy_score(y_test, attacker.predict(X_test)))
    
    return attacker











