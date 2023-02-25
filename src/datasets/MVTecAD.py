from torch.utils.data import Subset
from torchvision.datasets import ImageFolder
from base.torchvision_dataset import TorchvisionDataset

import numpy as np
import torch
import torchvision.transforms as transforms


class MVTecAD_Dataset(TorchvisionDataset):
    def __init__(self, root: str, normal_class: int = 0, data_augmentation: bool = False, normalize: bool = False):
        super(MVTecAD_Dataset, self).__init__(root)

        classes = ['defect', 'good']

        self.image_size = (3, 900, 900)

        # Define normal and outlier classes
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 2))
        self.outlier_classes.remove(normal_class)

        # ImageNet preprocessing: feature scaling to [0, 1], data normalization, and data augmentation
        train_transform = [transforms.Resize(224)]
        test_transform = [transforms.Resize(224)]

        if data_augmentation:
            # only augment training data
            train_transform += [transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
                                transforms.RandomHorizontalFlip(p=0.5)]
        train_transform += [transforms.ToTensor()]
        test_transform += [transforms.ToTensor()]

        if normalize:
            train_transform += [transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
            test_transform += [transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]

        train_transform = transforms.Compose(train_transform)
        test_transform = transforms.Compose(test_transform)
        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        # Get train set
        train_set = MyMVTecAD(root=self.root+'/MVTecAD/train', transform=train_transform, target_transform=target_transform)

        # Subset train_set to normal_classes
        idx = np.argwhere(np.isin(np.array(train_set.targets), self.normal_classes))
        idx = idx.flatten().tolist()
        train_set.semi_targets[idx] = torch.zeros(len(idx)).long()
        self.train_set = Subset(train_set, idx)

        # Get test set
        self.test_set = MyMVTecAD(root=self.root + '/MVTecAD/test', transform=train_transform, target_transform=target_transform)


# Get dataset
class MyMVTecAD(ImageFolder):
    def __init__(self, *args, **kwargs):
        super(MyMVTecAD, self).__init__(*args, **kwargs)

        self.semi_targets = torch.zeros(len(self.targets), dtype=torch.int64)

    def __getitem__(self, index):
        """
        Override the original method of the ImageFolder class.
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target, semi_target, index)
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        semi_target = int(self.semi_targets[index])

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, semi_target, index
