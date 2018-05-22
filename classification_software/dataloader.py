from __future__ import print_function, division
import os
import numpy as np


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import IMG_EXTENSIONS
from torch.utils.data.sampler import WeightedRandomSampler, RandomSampler
from torch.utils.data import DataLoader
from torchvision import transforms
import tensorflow as tf
import utils

def classification_data(datadir, train_size,val_size,test_size, BATCH_SIZE):
    #TODO: Turn this into a function and add arguments for data augementation.
    #def get_dataloaders(datadir,train_split,val_split, test_split, batch_size,)
    data_transforms = {
        'train': transforms.Compose([
            transforms.Scale(224),
            utils.RandomAffineTransform([0.9, 1.1], [0, np.pi], [-np.pi*5/180, np.pi*5/180], [0.05, 0.05]),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Scale(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Scale(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }


    ## This is because torch vision does not support tif files
    IMG_EXTENSIONS.append('tif')


    #get the main folder for our classification problem to feed from
    data = ImageFolder(datadir)

    #lengths to divide up the data into
    lengths = utils.create_lengths(data,train_size,val_size,test_size)

    # split the data into train, test, val
    names =['train','val','test']
    train_data, val_data, test_data = utils.random_split(data, lengths=lengths,transforms=data_transforms,names=names)


    # because we are splitting on the fly we need to calculate the weights in the training set to upsample by
    weights = utils.make_weights_for_balanced_classes(train_data.imgs, len(data.classes))
    weights = torch.DoubleTensor(weights)
    print("the train weights are {}".format(weights))
    print(len(weights))

    # create Samplers. For training we want to use weighted random samplers at test time we will use unweighted
    train_sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights))
    val_sampler = RandomSampler(val_data)


    # Create dataloaders
    train_loader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=False,
                              sampler=train_sampler,
                              drop_last=True)

    val_loader = DataLoader(dataset=val_data,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            sampler=val_sampler,
                            drop_last=True)

    test_loader = DataLoader(dataset=test_data,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            sampler=val_sampler,
                            drop_last=True)
    print('datasizes')
    dataset_sizes={'train':len(train_data),'val':len(val_data),'test':len(test_data)}
    dataloaders={'train':train_loader, 'val':val_loader, 'test':test_loader}

    return dataloaders, dataset_sizes

