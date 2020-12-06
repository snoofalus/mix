import numpy as np
from PIL import Image
import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

import torchvision
import torch
import torch.utils.data as data # only for checking dataloader, delete after
import torchvision.transforms as transforms

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

def get_s1s2glcm(root, n_labeled,
                 transform_train=None, transform_val=None,
                 download=True):

    norm = np.load(os.path.join(root,'workdir/norm.npy'))
    mu = norm[:,0].tolist()
    standard = norm[:,1].tolist()
    channel_stats = dict(mean=mu, std=standard)

    transform_train = transforms.Compose([
        RandomPadandCrop(32),
        RandomFlip(),
        ToTensor(),
        transforms.Normalize(**channel_stats),
    ])

    transform_val = transforms.Compose([
        ToTensor(),
        transforms.Normalize(**channel_stats),
    ])

    def npy_loader(path):
        #sample = torch.from_numpy(np.load(path))
        sample = np.load(path)
        return sample

    trainvaldir = os.path.join(root, 'images/s1s2glcm/by-image/train+val')
    testdir = os.path.join(root, 'images/s1s2glcm/by-image/test')

    trainval_dataset = torchvision.datasets.DatasetFolder(root=trainvaldir,loader=npy_loader,transform=None,extensions=('.npy'))
    train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(trainval_dataset.targets, int(n_labeled/6))
    train_labeled_dataset = S1S2GLCM_labeled(root=trainvaldir, indexs=train_labeled_idxs, loader=npy_loader,transform=transform_train,extensions=('.npy'))
    train_unlabeled_dataset = S1S2GLCM_unlabeled(root=trainvaldir, indexs=train_unlabeled_idxs, loader=npy_loader,transform=TransformTwice(transform_train),extensions=('.npy'))
    val_dataset = S1S2GLCM_labeled(root=trainvaldir, indexs=val_idxs, loader=npy_loader,transform=transform_val,extensions=('.npy'))
    test_dataset = torchvision.datasets.DatasetFolder(root=testdir,loader=npy_loader,transform=transform_val,extensions=('.npy'))

    print (f"#Labeled: {len(train_labeled_dataset)} #Unlabeled: {len(train_unlabeled_dataset)} #Val: {len(val_dataset)}")
    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset
    

def train_val_split(labels, n_labeled_per_class):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []
    n_val_per_class = int(len(labels)/(10*6))

    for i in range(0,5+1):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class:-n_val_per_class])
        val_idxs.extend(idxs[-n_val_per_class:])

    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs

cifar10_mean = (0.4914, 0.4822, 0.4465) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616) # equals np.std(train_set.train_data, axis=(0,1,2))/255

def normalise(x, mean=cifar10_mean, std=cifar10_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean*255
    x *= 1.0/(255*std)
    return x

def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target]) 

def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border)], mode='reflect')

class RandomPadandCrop(object):
    """Crop randomly the image.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, x):
        x = pad(x, 4)

        h, w = x.shape[1:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        x = x[:, top: top + new_h, left: left + new_w]

        return x

class RandomFlip(object):
    """Flip randomly the image.
    """
    def __call__(self, x):
        if np.random.rand() < 0.5:
            x = x[:, :, ::-1]

        return x.copy()

class GaussianNoise(object):
    """Add gaussian noise to the image.
    """
    def __call__(self, x):
        c, h, w = x.shape
        x += np.random.randn(c, h, w) * 0.15
        return x

class ToTensor(object):
    """Transform the image to tensor.
    """
    def __call__(self, x):
        x = torch.from_numpy(x)
        return x

class S1S2GLCM_labeled(torchvision.datasets.DatasetFolder):
    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            indexs=None
    ) -> None:
        super(S1S2GLCM_labeled, self).__init__(root, loader=loader, extensions=extensions, transform=transform,
                                            target_transform=target_transform, is_valid_file=is_valid_file)
        if indexs is not None:
            new_samples = [self.samples[i] for i in indexs]
            self.samples=new_samples


            #self.targets = np.array(self.targets)[indexs]
            new_targets = [self.targets[i] for i in indexs]
            self.targets = new_targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

class S1S2GLCM_unlabeled(S1S2GLCM_labeled):
    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            indexs=None
    ) -> None:
        super(S1S2GLCM_unlabeled, self).__init__(root, loader=loader, extensions=extensions, transform=transform,
                                            target_transform=target_transform, is_valid_file=is_valid_file, indexs=indexs)
        #self.targets = np.array([-1 for i in range(len(self.targets))])
        self.targets = [-1 for i in range(len(self.targets))]

class CIFAR10_labeled(torchvision.datasets.CIFAR10):

    def __init__(self, root, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_labeled, self).__init__(root, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = self.targets[indexs]
            #self.targets = np.array(self.targets)[indexs]
        self.data = transpose(normalise(self.data)) #not needed for s1s2glcm

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    

class CIFAR10_unlabeled(CIFAR10_labeled):

    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_unlabeled, self).__init__(root, indexs, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        self.targets = np.array([-1 for i in range(len(self.targets))])
        