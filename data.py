import torch
import torchvision
from torch.utils.data import DataLoader
import wandb

CODE_TEST = False

def getTransforms():
    return torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                           torchvision.transforms.ToTensor()])



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


def getDataLoaders(t_bs, v_bs, r_bs):
    if wandb.config.db == "MNIST":
        db = torchvision.datasets.MNIST
    elif wandb.config.db == "CIFAR10":
        db = torchvision.datasets.CIFAR10
    else:
        raise NotImplementedError("Dataset " + str(db) + " is not implemented")
    data = db(root = './dataset/',
              train = True, download = True,
              transform = getTransforms())
    if CODE_TEST:
        subset = list(range(0,len(data), int(len(data)/100)))
        data = torch.utils.data.Subset(data, subset)

    train_loader = torch.utils.data.DataLoader(data,
                                              batch_size = t_bs,
                                              shuffle = True,
                                              num_workers = 4)

    data = db(root = './dataset/',
              train = False, download = True,
              transform = getTransforms())
    validate_loader = torch.utils.data.DataLoader(data,
                                                  batch_size = v_bs,
                                                  shuffle = True,
                                                  num_workers = 4)

    data = db(root = './dataset/',
              train = True, download = True,
              transform = getTransforms())
    ref_loader = torch.utils.data.DataLoader(data,
                                             batch_size = r_bs,
                                             shuffle = True,
                                             num_workers = 4)
     
    return train_loader, validate_loader, ref_loader
