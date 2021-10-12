# https://github.com/pytorch/vision/blob/master/torchvision/models
import torch
import torch.nn as nn
from collections import OrderedDict
from nets.base_models import MyNetwork
from utils.get_data_iter import cutout_batch


cifar_cfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


class VGG_CIFAR(MyNetwork):
    def __init__(self, cfg=None, cutout=True, num_classes=10):
        super(VGG_CIFAR, self).__init__()
        if cfg is None:
            cfg = cifar_cfg[16]
        self.cutout = cutout
        self.cfg = cfg
        _cfg = list(cfg)
        self._cfg = _cfg
        self.feature = self.make_layers(_cfg, True)
        self.avgpool = nn.AvgPool2d(2)
        self.classifier = nn.Sequential(
            nn.Linear(self.cfg[-1], 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )
        self.num_classes = num_classes
        self.classifier_param = (
            self.cfg[-1] + 1) * 512 + (512 + 1) * num_classes

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        pool_index = 0
        conv_index = 0
        for v in cfg:
            if v == 'M':
                layers += [('maxpool_%d' % pool_index,
                            nn.MaxPool2d(kernel_size=2, stride=2))]
                pool_index += 1
            else:
                conv2d = nn.Conv2d(
                    in_channels, v, kernel_size=3, padding=1, bias=False)
                conv_index += 1
                if batch_norm:
                    bn = nn.BatchNorm2d(v)
                    layers += [('conv_%d' % conv_index, conv2d), ('bn_%d' % conv_index, bn),
                               ('relu_%d' % conv_index, nn.ReLU(inplace=True))]
                else:
                    layers += [('conv_%d' % conv_index, conv2d),
                               ('relu_%d' % conv_index, nn.ReLU(inplace=True))]
                in_channels = v
        self.conv_num = conv_index
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        if self.training and self.cutout:
            with torch.no_grad():
                x = cutout_batch(x, 16)
        x = self.feature(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

    def feature_extract(self, x):
        tensor = []
        for _layer in self.feature:
            x = _layer(x)
            if type(_layer) is nn.ReLU:
                tensor.append(x)
        return tensor

    @property
    def config(self):
        return {
            'name': self.__class__.__name__,
            'cfg': self.cfg,
            'cfg_base': [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512],
            'dataset': 'cifar10',
        }


def test():
    net = VGG_CIFAR()
    tensor_input = torch.randn([2, 3, 32, 32])
    feature = net.feature_extract(tensor_input)
    # import pdb; pdb.set_trace()
    return feature
    pass

def VGG16():
    return VGG_CIFAR()
# if __name__ == "__main__":
#     test()
