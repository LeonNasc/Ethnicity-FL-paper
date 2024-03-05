import torchvision.transforms as transforms
import numpy as np
import glob
import os
import torch
import random

from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision.datasets import CIFAR10, CIFAR100, CelebA
from typing import Any, Callable, Optional, Tuple

BATCH_SIZE = 64

def partition_CIFAR_IID(num_clients, CIFAR_TYPE="CIFAR10"):
    trainset, testset = load_CIFAR(CIFAR_TYPE)
    #returns a tuple of (trainloaders, valloaders, testloaders)
    return IID_setup(num_clients=num_clients, trainset=trainset, testset=testset)


def partition_FedFaces_IID(num_clients, undersample=True):
    train, test = load_fedfaces(undersample=undersample)
    #returns a tuple of (trainloaders, valloaders, testloaders)
    return IID_setup(num_clients=num_clients, trainset=train, testset=test)


def partition_CIFAR_nonIID(num_clients, CIFAR_TYPE="CIFAR10", beta=0.5):
    train, test = load_CIFAR(CIFAR_TYPE)
    #returns a tuple of (trainloaders, valloaders, testloaders)
    return nonIID_setup(num_clients, beta, train, test)

def partition_FedFaces_nonIID(num_clients, beta=0.5):
    train, test = load_fedfaces()
    #returns a tuple of (trainloaders, valloaders, testloaders)
    return nonIID_setup(num_clients, beta, train, test)

def partition_CelebA_IID(num_clients):
    trainset,testset = load_CelebA()
    return IID_setup(num_clients=num_clients, trainset=trainset, testset=testset)

def partition_CelebA_nonIID(num_clients, beta=0.5):
    trainset,testset = load_CelebA()
    return nonIID_setup(num_clients, beta, trainset, testset)
'''
For the function below, I adapted a data partition strategy proposed by Li et al in the study

Federated Learning on Non-IID Data Silos: An Experimental Study

Found @: https://arxiv.org/pdf/2102.02079.pdf

'''
def IID_setup(num_clients, trainset, testset):
    partition_size = 1 / num_clients
    lengths = [partition_size] * num_clients
    datasets = random_split(trainset, lengths)
    # Split each partition into train/val and create DataLoader
    trainloaders, valloaders, testloader = make_loaders(num_clients, testset, datasets)
    return trainloaders,valloaders,testloader


def nonIID_setup(num_clients, beta, train, test):
    print(f'Shape nonIID: {train.data[0].shape}')
    datasets = _dirilecht_partition_data(train.data, train.targets, num_clients, beta)
    trainloaders, valloaders, testloader = make_loaders(num_clients, test, datasets)
    return trainloaders,valloaders,testloader

def convert_celebA_to_numpy(split, split_name, save=False):
    print(f"Converting {split_name} into npz")

    xs = []
    ys = []

    for i in range(1,len(split)):
        x, y = split[i]
        xs.append(x) 
        ys.append(y)

    xs = np.array(xs)
    ys = np.array(ys)

    if save:
        np.savez(f"./Datasets/ready/celebA/loaded_np/{split_name}", xs=xs, ys=ys)

    print("Finished")
    return xs,ys

def _dirilecht_partition_data(xs, ys, n_parties, beta):

    # Partition mini-batch sizes
    min_size = 0                #lower boundary
    min_require_size = 10       #upper boundary

    N = len(ys)
    ys = np.array(ys)
    net_dataidx_map = {}

    while min_size < min_require_size: #Makes many small groups of up to 10 members of a given class

        idx_batch = [[] for _ in range(n_parties)]

        for k in list(set(ys)):
            idx_k = np.where(ys == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))

            ## Balance
            proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]

            min_size = min([len(idx_j) for idx_j in idx_batch])


    for j in range(n_parties):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]

    partitioned_data = []
    for k,v in net_dataidx_map.items():
        if len(xs.shape) == 3:
            channels = 1
        else :
            channels = 3

        print(xs.shape)
        corr_shape = (len(v),channels, xs.shape[-2],xs.shape[-1])
        local_xs = torch.Tensor(xs[v].astype(np.float32).reshape(corr_shape))
        local_ys = torch.LongTensor(ys[v].astype(np.float32))
        partitioned_data.append(TensorDataset(local_xs,local_ys))

    return partitioned_data



def make_loaders(num_clients, testset, datasets):
    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = len(ds) // num_clients 
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths)
        trainloaders.append(DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=BATCH_SIZE))
        testloader = DataLoader(testset, batch_size=BATCH_SIZE)

    return trainloaders,valloaders,testloader


'''
Data Loaders
'''
def load_CIFAR(CIFAR_TYPE):
    # Download and transform CIFAR-10 (train and test)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    
    if CIFAR_TYPE == "CIFAR10":
        trainset = CIFAR10("./Datasets/ready/cifar-10-python", train=True, transform=transform)
        testset = CIFAR10("./Datasets/ready/cifar-10-python", train=False, transform=transform)
    else:
        trainset = CIFAR100("./Datasets/ready/cifar-100-python", train=True, transform=transform)
        testset = CIFAR100("./Datasets/ready/cifar-100-python", train=False, transform=transform)

    return trainset,testset

def load_CelebA():
    # Download and transform CIFAR-10 (train and test)
    globed =glob.glob("./Datasets/ready/celebA/loaded_np/*.npz")
    if  len(globed) == 0:
        transform = transforms.Compose(
            [transforms.Resize((64,64)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        target_transform = transforms.Compose(
            [lambda y: y[2]]
        )

        #For the experiment, we're using the attribute attractive
        attr = np.zeros(40)
        attr[2] = 1

        attr = torch.Tensor(attr)
        
        trainset = CelebA("./Datasets/ready/celebA/", split="valid", target_type="attr", download=False, transform=transform, target_transform=target_transform)
        testset = CelebA("./Datasets/ready/celebA/", split="test", target_type="attr", download=False, transform=transform, target_transform= target_transform)

        convert_celebA_to_numpy(trainset, "valid", True)
        convert_celebA_to_numpy(testset, "test", True)
    else:
        print("Loaded from NPZ")
        train = np.load("./Datasets/ready/celebA/loaded_np/valid.npz")
        trainset = LoadedCelebA(train["xs"], train["ys"])
        test = np.load("./Datasets/ready/celebA/loaded_np/test.npz")
        testset = LoadedCelebA(test["xs"], test["ys"])

    return trainset,testset


'''
 FedFaces data is very unbalanced, so we'll be utilzing undersampling to organize the data.
 Also, some utilities classes are implemented for compatibility with Torch
'''
def convert_to_3d_with_repeat(array_2d):
    """
    Convert a 2D numpy array to a 3D array with values repeated along the new dimension.

    Parameters:
    array_2d (numpy.ndarray): The input 2D numpy array.

    Returns:
    numpy.ndarray: The converted 3D numpy array with shape (N, M, 3).
    """
    # Add a new axis to the 2D array
    array_3d = array_2d[:, :, np.newaxis]
    
    # Repeat the values along the new dimension
    array_3d = np.repeat(array_3d, 3, axis=2)
    
    return array_3d


def load_fedfaces(undersample=True):
    data_all = [np.load(f) for f in glob.glob("./Datasets/ready/prepared_data/regions/*.npz")]

    merged_data = {"X": [convert_to_3d_with_repeat(x) for x in data_all[0]["X"]], "Y": data_all[0]["Y"]} 

    for data in data_all[1:]:
        xs = np.concatenate([merged_data["X"], [convert_to_3d_with_repeat(x) for x in data["X"]]])
        ys = np.concatenate([merged_data["Y"], data["Y"]])
        merged_data["X"] = xs
        merged_data["Y"] = ys

    if undersample:
       merged_data = undersample_fedfaces(merged_data)


    msk = np.random.rand(len(merged_data["X"])) < 0.8

    train_x, train_y = (merged_data["X"][msk], merged_data["Y"][msk])
    test_x, test_y = (merged_data["X"][~msk], merged_data["Y"][~msk])

    train = FedFaces(train_x.astype(np.float32),train_y)
    test = FedFaces(test_x.astype(np.float32), test_y)

    return train, test

def undersample_fedfaces(merged_data):
    xs, ys = ([], [])

    class_1 = np.where(merged_data["Y"].astype(int) == 1)[0]
    class_2 = np.where(merged_data["Y"].astype(int) == 2)[0]
    class_3 = np.where(merged_data["Y"].astype(int) == 3)[0]

    minority = min(len(class_1), len(class_2), len(class_3))
    print(f'Minority class count: {minority}')
    for _class in [class_1, class_2, class_3]:
        choices = random.choices(_class, k=minority)
        x = merged_data["X"][choices]
        y = merged_data["Y"][choices]

        xs.append(x)
        ys.append(y)
        
    data = {}
    data["X"] = np.concatenate([xs[0], xs[1], xs[2]])
    data["Y"] = np.concatenate([ys[0], ys[1], ys[2]])

    return data


class FedFaces(torch.utils.data.Dataset):

    def __init__(self, 
                    xs, 
                    ys,
                    transform: Optional[Callable] = None,
                    target_transform: Optional[Callable] = None):

        self.data = xs
        self.targets = ys

        self.data = self.data.transpose((0, 3, 1, 2))  # convert to HWC

        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
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


    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        return True
        


class LoadedCelebA(torch.utils.data.Dataset):

    def __init__(self, 
                    xs, 
                    ys,
                    transform: Optional[Callable] = None,
                    target_transform: Optional[Callable] = None):

        self.data = xs
        self.targets = ys
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
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


    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        return True