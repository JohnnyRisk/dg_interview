import random
from math import floor
import itertools
from torch._utils import _accumulate
from torch import randperm
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.transform import warp, AffineTransform


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.transform import warp, AffineTransform


class RandomAffineTransform(object):
    def __init__(self,
                 scale_range,
                 rotation_range,
                 shear_range,
                 translation_range
                 ):
        self.scale_range = scale_range
        self.rotation_range = rotation_range
        self.shear_range = shear_range
        self.translation_range = translation_range

    def __call__(self, img):
        img_data = np.array(img)
        h, w, n_chan = img_data.shape
        scale_x = np.random.uniform(*self.scale_range)
        scale_y = np.random.uniform(*self.scale_range)
        scale = (scale_x, scale_y)
        rotation = np.random.uniform(*self.rotation_range)
        shear = np.random.uniform(*self.shear_range)
        translation = (
            np.random.uniform(*self.translation_range) * w,
            np.random.uniform(*self.translation_range) * h
        )
        af = AffineTransform(scale=scale, shear=shear, rotation=rotation, translation=translation)
        img_data1 = warp(img_data, af.inverse)
        img1 = Image.fromarray(np.uint8(img_data1 * 255))
        return img1

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class Subset(ImageFolder):
    def __init__(self, dataset, indices, transform=None, target_transform=None, loader=default_loader):
        self.dataset = dataset
        print(indices.numpy().shape)
        self.indices = indices.numpy()
        self.classes = dataset.classes
        self.transform = transform
        self.target_transform = target_transform
        self.class_to_idx = dataset.class_to_idx
        self.imgs = np.array(dataset.imgs)[self.indices]
        self.loader = loader


    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, int(target)

    def __len__(self):
        return len(self.indices)


def create_lengths(dataset,train,val,test):
    assert train+test+val >= 0.9999
    assert train+test+val <= 1.0001
    length = len(dataset)
    trn = floor(train*length)
    val = floor(val *length)
    test = length -trn - val
    assert length == trn+val+test
    return [trn,val,test]

def stratefied_train_valid_split(dataset, test_size=0.20, shuffle=False, random_seed=0):
    """ Return a list of splitted indices from a DataSet in a stratefied fashion.
    Indices can be used with DataLoader to build a train and validation set.

    Arguments:
        A Dataset
        A test_size, as a float between 0 and 1 (percentage split) or as an int (fixed number split)
        Shuffling True or False
        Random seed
    """
    # get the length of the dataset and the number of classes
    length = dataset.__len__()
    n_classes = len(dataset.classes)
    indices = list(0, length-1)
    train_indices=[]
    test_indices=[]

    # run through the number of classes and look at the class label which is the second element in the tuple
    # get the length of this and then add it to the length class. Then update the start indicies by looking at the
    # previous start indicies and the length of the previous class.
    for i in range(n_classes):
        selector = [x for x in dataset if x[1]==i]
        class_indices = list(itertools.compress(indices, selectors=selector))
        length_class = len(class_indicies)

        if shuffle == True:
            random.seed(random_seed)
            random.shuffle(class_indices)

        if type(test_size) is float:
            split = floor(test_size * length_class)
        elif type(test_size) is int:
            split = test_size
        else:
            raise ValueError('%s should be an int or a float' % str)
        train_indices += class_indices[split:]
        test_indices += class_indices[:split]
    return train_indices, test_indices


def get_weights(dataset):
    """
    We get the weights as the inverse of the amount of times that they appear in the the dataset
    This helps us to upsample the more sparse classes so they occur more often. It will come out so
    that they are sampled on average a uniform distribution for each class.
    :param dataset:
    :return: weights
    """
    print('getting weights')
    length=len(dataset)
    n_classes = len(dataset.classes)
    count=[0]*n_classes
    weights = []
    print('length of dataset is {}'.format(length))
    for i in range(length):
        count[int(dataset.imgs[i,1])] +=1
    print('done counting')
    N = sum(count)
    print(count)
    print(N)
    for i in range(n_classes):
        weights.append(N/count[i])
    return weights

def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[int(item[1])] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[int(val[1])]
    return weight

def random_split(dataset, lengths,transforms=None,names=None):
    """
    Randomly split a dataset into non-overlapping new datasets of given lengths
    ds

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (iterable): lengths of splits to be produced
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")
    indices = randperm(sum(lengths))

    if transforms:
        if len(transforms) != len(names):
            raise ValueError("Number of transforms has to be the same as number of names")
        if len(lengths) != len(transforms):
            raise ValueError("Must have equal number of transforms as datset partitions")
        return [Subset(dataset, indices[offset - length:offset], transform=transforms[name]) for offset, length, name in zip(_accumulate(lengths), lengths, names)]

    return [Subset(dataset, indices[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]