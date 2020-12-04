# -*- coding:utf8 -*-
# @TIME     : 2020/10/30 16:07
# @Author   : Hao Su
# @File     : dataload.py

import torchvision.transforms as transforms
import torchvision.datasets as dset


def load_dataset(dataroot, dataset_name, imageSize, trans, train=True):
    params_med = {"dataroot": dataroot, "split": 'train' if train else 'test', "transform":trans}
    if dataset_name == 'mnist':
        dataset = dset.MNIST(root=dataroot,
                             train=train,
                             download=True,
                             transform=trans)
    return dataset
