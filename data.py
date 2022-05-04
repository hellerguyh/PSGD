import torch
import torchvision
from torch.utils.data import DataLoader
import wandb

from PIL import Image
import numpy as np

CODE_TEST = False 

def getTransforms():
    return torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                           torchvision.transforms.ToTensor()])


nnType2DsName = {
    'LeNet5'    : 'MNIST',
    'ResNet18'  : 'CIFAR10',
    'ResNet18-100'  : 'CIFAR100',
}

DatasetMap = {
              'MNIST': torchvision.datasets.MNIST,
              'CIFAR10' : torchvision.datasets.CIFAR10,
              'CIFAR100' : torchvision.datasets.CIFAR100
             }


def getTransforms():
    return torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                           torchvision.transforms.ToTensor()])

'''
getDL - gets a dataloader without going through wandb.config
@bs: block size
@train: if True it takes the train dataset
@ds_name: MNIST/CIFAR10/...
'''
def getDL(bs, train, ds_name):
    db = DatasetMap[ds_name]
    data = db(root = './dataset/',
              train = train, download = True,
              transform = getTransforms())

    if CODE_TEST:
        subset = list(range(0,len(data), int(len(data)/1000)))
        data = torch.utils.data.Subset(data, subset)

    if ds_name == "MNIST":
        torch.set_num_threads(4)
        NW = 4
    else:
        NW = 4

    loader = torch.utils.data.DataLoader(data, batch_size = bs, shuffle = True,
                                         num_workers = NW, pin_memory = True)

    return loader, len(data)

'''
Iterator that starts again upon StopIteration
'''
def getULItr(iterable_obj):
    itr = iter(iterable_obj)
    while True:
        try:
            yield next(itr)
        except StopIteration:
            itr = iter(iterable_obj)
            yield next(itr)
