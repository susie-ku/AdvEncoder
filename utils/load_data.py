import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np

import enum
import os
import torch

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset

from functools import partial
from sklearn.model_selection import train_test_split
torch.manual_seed(42)

from torchvision.models import (
    # densenet121,
    densenet161,
    efficientnet_b0, 
    efficientnet_b3,
    inception_v3,
    resnet101,
    resnet152,
    # resnet50,
    vgg19,
    wide_resnet101_2,
    # wide_resnet50_2
)

from torchvision.models import (
    # DenseNet121_Weights,
    DenseNet161_Weights,
    EfficientNet_B0_Weights,
    EfficientNet_B3_Weights,
    Inception_V3_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
    # ResNet50_Weights,
    VGG19_Weights,
    Wide_ResNet101_2_Weights,
    # Wide_ResNet50_2_Weights
)

@enum.unique
class Datasets(str, enum.Enum):
    CIFAR10 = "CIFAR10"
    CIFAR100 = "CIFAR100"
    ImageNet = "ImageNet"


def vit_features(image, feature_extractor):
    return feature_extractor(
            images=image, 
            return_tensors="pt"
        )['pixel_values'].squeeze(0)


def get_vit_transforms(feature_extractor):
    return transforms.Compose([
        transforms.ToTensor(),
        partial(vit_features, feature_extractor=feature_extractor)
    ])

def get_dataset(path, dataset_name, transform=None):
    # path = os.path.join(path_to_data, dataset_name)
    if dataset_name == Datasets.CIFAR10:
        dataset = datasets.CIFAR10(path, train=False, download=True, transform=transform)
    elif dataset_name == Datasets.CIFAR100:
        dataset = datasets.CIFAR100(path, train=False, download=True, transform=transform)
    elif dataset_name == Datasets.ImageNet:
        dataset = datasets.ImageFolder(path, transform=transform)
    else:
        raise NotImplementedError()
    
    return dataset

class IndexedDataset(Dataset):
    def __init__(self, dataset, transform=None):
        super().__init__()
        self.dataset = dataset
        self.transform = transform
        # self.targets = [s[1] for s in self.dataset]
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if self.transform:
            idx, image, label = self.dataset[idx]
            image = self.transform(image)
        else:
            image, label = self.dataset[idx]
        return idx, image, label

class TransformerIndexedDataset(Dataset):
    def __init__(self, dataset, transform=None):
        super().__init__()
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.transform:
            idx, image, label = self.dataset[idx]
            image = self.dataset.transform(image)
        else:
            image, label = self.dataset[idx]
        return idx, image, label

def load_data(data, bs):

    transform_ = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

    if data == 'cifar10':
        train_dataset = datasets.CIFAR10('../Dataset/data/cifar10/', train=True, download=False, transform=transform_)
        test_dataset = datasets.CIFAR10('../Dataset/data/cifar10/', train=False, download=False, transform=transform_)

    elif data == 'stl10':
        train_dataset = datasets.STL10('../Dataset/data/stl10', split="train", download=True, transform=transform_)
        test_dataset = datasets.STL10('../Dataset/data/stl10', split="test", download=True, transform=transform_)

    elif data == 'gtsrb':
        train_dataset = datasets.ImageFolder('../Dataset/GTSRB/Train/', transform = transform_)
        train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [30000, len(train_dataset)-30000])

    elif data == 'imagenet':
        dataset = IndexedDataset(
            get_dataset(
                '/media/ssd-3t/kkuvshinova/ImageNet',
                Datasets.ImageNet
            )
        )

        train_dataset_, dataset_ = torch.utils.data.random_split(
            dataset,
            [256, len(dataset) - 256],
            generator=torch.Generator().manual_seed(42)
        )
        train_dataset = IndexedDataset(
            train_dataset_, transform=DenseNet161_Weights.IMAGENET1K_V1.transforms()
        )
        eval_dataset = IndexedDataset(
            dataset_, transform=DenseNet161_Weights.IMAGENET1K_V1.transforms()
        )

        print('targets computation begins')
        # targets = np.array([items[2] for items in eval_dataset])
        # np.save('targets.npy', targets)
        # print(targets)
        # targets = dataset.targets[eval_dataset.indices]
        with open('targets.npy', 'rb') as f:
            targets = np.load(f).tolist()
        print('targets computation ends')

        val_idx, test_idx = train_test_split(
            np.arange(len(targets)),
            test_size=len(eval_dataset) - 5000,
            shuffle=True,
            stratify=targets,
            random_state=42
        )

    val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)

        
    print('Train dataset: %d, Test dataset: %d'%(len(train_dataset),len(dataset_)))
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=False
    )
    test_loader = DataLoader(
        dataset=eval_dataset,
        sampler=test_sampler,
        batch_size=64,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=False
    )

    # train_loader = DataLoader(train_dataset, batch_size=bs, num_workers=5, drop_last=False, shuffle=True)
    # test_loader  = DataLoader(test_dataset, batch_size=bs, num_workers=5, drop_last=False, shuffle = False)

    return train_loader,  test_loader



def normalzie(args, x):

    if args.dataset == 'cifar10':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        normalizer = transforms.Normalize(mean=mean, std=std)
        return normalizer(x)

    elif args.dataset == 'stl10':
        mean = (0.44087798, 0.42790666, 0.38678814)
        std = (0.25507198, 0.24801506, 0.25641308)
        normalizer = transforms.Normalize(mean=mean, std=std)
        return normalizer(x)

    elif args.dataset == 'gtsrb':
        mean = (0.44087798, 0.42790666, 0.38678814)
        std = (0.25507198, 0.24801506, 0.25641308)
        normalizer = transforms.Normalize(mean=mean, std=std)
        return normalizer(x)

    else:
        return x



