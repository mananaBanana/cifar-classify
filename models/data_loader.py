import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class CIFARLoader():
    def __init__(self, dataset_type='CIFAR10', root='./data/CIFAR10', sample=False, transform_list=[transforms.ToTensor()]):
        self.transform_list = transform_list
        self.dataloader = {}
        self.classlist = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.sample = sample

        if dataset_type == 'CIFAR10':
            self.train_set = torchvision.datasets.CIFAR10(
                                root=root,
                                train=True,
                                download=True,
                                transform=transforms.Compose(self.transform_list)
                                )
            self.test_set = torchvision.datasets.CIFAR10(
                                root=root,
                                train=False,
                                download=True,
                                transform=transforms.Compose(self.transform_list)
                                )

    def get_dataloader(self, batch_size):
        """ 
        Get train/test dataloader based on list 
        specified by splits
        """
        for split in ['train', 'test']:
            if split == 'train':
                if self.sample:
                    self.dataloader[split] = DataLoader(self.train_set, batch_size=batch_size,
                                                        sampler=SubsetRandomSampler(list(range(1000))))
                else:
                    self.dataloader[split] = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)
            if split == 'test':
                if self.sample:
                    self.dataloader[split] = DataLoader(self.test_set, batch_size=batch_size,
                                                        sampler=SubsetRandomSampler(list(range(1000))))
                else:
                    self.dataloader[split] = DataLoader(self.test_set, batch_size=batch_size, shuffle=True)

        return self.dataloader

    def get_class_list(self):
        return self.classlist
