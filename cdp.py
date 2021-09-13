import copy
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from nni.algorithms.compression.pytorch.pruning \
    import L1FilterPruner, L1FilterPrunerMasker
from utils.counter import count_flops_params

class TFIDFMasker(L1FilterPrunerMasker):
    def __init__(self, model, pruner, threshold, tf_idf_map, preserve_round=1, dependency_aware=False):
        super().__init__(model, pruner, preserve_round, dependency_aware)
        self.threshold=threshold
        self.tf_idf_map=tf_idf_map
        
    def get_mask(self, base_mask, weight, num_prune, wrapper, wrapper_idx, channel_masks=None):
        # get the l1-norm sum for each filter
        w_tf_idf_structured = self.get_tf_idf_mask(wrapper, wrapper_idx)
        
        mask_weight = torch.gt(w_tf_idf_structured, self.threshold)[
            :, None, None, None].expand_as(weight).type_as(weight)
        mask_bias = torch.gt(w_tf_idf_structured, self.threshold).type_as(
            weight).detach() if base_mask['bias_mask'] is not None else None

        return {'weight_mask': mask_weight.detach(), 'bias_mask': mask_bias}
    
    def get_tf_idf_mask(self, wrapper, wrapper_idx):
        name = wrapper.name
        if wrapper.name.split('.')[-1] == 'module':
            name = wrapper.name[0:-7]
        #print(name)
        w_tf_idf_structured = self.tf_idf_map[name]
        return w_tf_idf_structured


class TFIDFPruner(L1FilterPruner):
    def __init__(self, model, config_list, cdp_config:dict, pruning_algorithm='l1', optimizer=None, **algo_kwargs):
        super().__init__(model, config_list, optimizer)
        self.set_wrappers_attribute("if_calculated", False)
        self.masker = TFIDFMasker(model, self, threshold=cdp_config["threshold"], tf_idf_map=cdp_config["map"], **algo_kwargs)
    def update_masker(self,model,threshold,mapper):
        self.masker = TFIDFMasker(model, self, threshold=threshold, tf_idf_map=mapper)        

def acculumate_feature(model, loader, stop:int):
    model=model.cuda()
    features = {}
    classes = []
    
    def hook_func(m, x, y, name, feature_iit):
        #print(name, y.shape) # ([256, 64, 8, 8])
        '''ReLU'''
        f = F.relu(y)    
        #f = y
        '''Average Pool'''
        feature = F.avg_pool2d(f, f.size()[3])
        #print(feature.shape) # ([256, 64, 1, 1])
        feature = feature.view(f.size()[0], -1)
        #print(feature.shape) # ([256, 64])
        feature = feature.transpose(0, 1)
        #print(feature.shape) 
        if name not in feature_iit:
            feature_iit[name] = feature.cpu()
        else:
            feature_iit[name] = torch.cat([feature_iit[name], feature.cpu()], 1)
            
    hook=functools.partial(hook_func, feature_iit=features)
    
    handler_list=[]
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
        #if not isinstance(m, nn.Linear):
            handler = m.register_forward_hook(functools.partial(hook, name=name))
            handler_list.append(handler)
    for batch_idx, (inputs, targets) in enumerate(loader):
        if batch_idx >= stop:
            break
        #if batch_idx % (10) == 0:
        print('batch_idx', batch_idx)
        model.eval()
        classes.extend(targets.numpy())
        with torch.no_grad():
            model(inputs.cuda())
    [ k.remove() for k in handler_list]
    '''Image-wise Activation'''
    return features, classes


def calculate_cdp(features:dict, classes:list, dataset:str, coe:int, unlearn_class:int):
    #print(len(classes))
    features_class_wise = {}
    tf_idf_map = {}
    if dataset == 'cifar10':
        class_num = 10
    list_classes_address = []
    for z in range(class_num):
        address_index = [x for x in range(len(classes)) if classes[x] == z]
        list_classes_address.append([z, address_index])
    dict_address = dict(list_classes_address)
    for fea in features:
        '''Class-wise Activation'''
        class_wise_features = torch.zeros(class_num, features[fea].shape[0])
        image_wise_features = features[fea].transpose(0, 1)
        for i, v in dict_address.items():
            for j in v:
                class_wise_features[i] += image_wise_features[j]    
            if len(v) == 0:
                class_wise_features[i] = 0
            else:
                class_wise_features[i] = class_wise_features[i] / len(v)
        features_class_wise[fea] = class_wise_features.transpose(0, 1)
        #print(features_class_wise[fea].shape) # ([64, 10])
        '''TF-IDF'''
        calc_tf_idf(features_class_wise[fea], fea, coe=coe, unlearn_class=unlearn_class, tf_idf_map=tf_idf_map)
        #print(tf_idf_map[fea].shape)
    return tf_idf_map
        
    
                                                                                  # c - filters; n - classes
def calc_tf_idf(feature, name:str, coe:int, unlearn_class:int, tf_idf_map:dict):  # feature = [c, n] ([64, 10])    
    # calc tf for filters
    sum_on_filters = feature.sum(dim=0)
    #print(feature_sum.shape) # ([10])
    balance_coe = np.log((feature.shape[0]/coe)*np.e) if coe else 1.0
    #print(feature.shape, name, coe)
    tf = (feature / sum_on_filters) * balance_coe
    #print(tf.shape) # ([64, 10])
    tf_unlearn_class = tf.transpose(0,1)[unlearn_class]
    #print(tf_unlearn_class.shape)
    
    # calc idf for filters
    classes_quant = float(feature.shape[1])
    mean_on_classes = feature.mean(dim=1).view(feature.shape[0], 1)
    #print(mean_on_classes.shape) # ([64, 1])
    inverse_on_classes = (feature >= mean_on_classes).sum(dim=1).type(torch.FloatTensor)
    #print(inverse_on_classes.shape) # ([64])
    idf = torch.log(classes_quant / (inverse_on_classes + 1.0))
    #print(idf.shape) # ([64])
    
    importance = tf_unlearn_class * idf
    #print(importance.shape) # ([64])
    tf_idf_map[name] = importance


def get_threshold_by_sparsity(mapper:dict, sparsity:float):
    assert 0<sparsity<1
    tf_idf_array=torch.cat([v for v in mapper.values()],0)
    threshold = torch.topk(tf_idf_array, int(tf_idf_array.shape[0]*(1-sparsity)))[0].min()
    return threshold

def get_threshold_by_flops(mapper:dict, reduced_ratio:float, rnet):
    pass
    # 二分查找最优阈值
    sparsity=reduced_ratio # use reduced_ratio as init sparsity
    tf_idf_array=torch.cat([v for v in mapper.values()],0)
    threshold = torch.topk(tf_idf_array, int(tf_idf_array.shape[0]*(1-sparsity)))[0].min()
    
    #cdp_config={"threshold": threshold, "map": mapper }
    config_list = [{'sparsity': sparsity,'op_types': ['Conv2d']}]

    ratio=0
    upper=tf_idf_array.shape[0]
    mid=int(tf_idf_array.shape[0]*(1-sparsity))
    lower=0
    count=0
    flops_r, param, detail_flops = count_flops_params(rnet, (1, 3, 32, 32))

    while(np.abs(ratio-reduced_ratio)> 0.003 and count<4):
        # 只要差距大于 0.5%
        # 如果按sparsity 得到的flops 被压缩率比目标值小 说明保留的filter多 则保留 小侧的区间 
        # 如果按sparsity 得到的flops 被压缩率比目标值大 说明保留的filter少 则保留 大侧的区间
        threshold = torch.topk(tf_idf_array, mid)[0].min()
        net=copy.deepcopy(rnet)
        pruner = TFIDFPruner(net, config_list, {"threshold": threshold, "map": mapper })
        _ = pruner.compress()
        flops, param, detail_flops = count_flops_params(net, (1, 3, 32, 32),verbose=False)
        ratio=(flops_r-flops)/flops_r
        if(ratio < reduced_ratio):
            upper=mid
        else:
            lower=mid
        mid=(upper+lower)//2
        count+=1
        print("Cutter Flops is: ",flops)
        print("Rate is: ",ratio)
    return threshold