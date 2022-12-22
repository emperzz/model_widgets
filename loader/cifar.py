from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms


def get_cifar_dataloader(cifar, root, batch_size, num_workers):
    cifar = cifar.lower()
    assert cifar in ('cifar10', 'cifar100')
    kwargs = {
        'root' : root,
        'download' : True,
        'transform' : transforms.Compose([transforms.Resize(64),
                                    transforms.ToTensor()])
    }
    trainset = CIFAR10(train = True, **kwargs) if cifar == 'cifar10' else CIFAR100(train = True, **kwargs)
    testset = CIFAR10(train = False, **kwargs) if cifar == 'cifar10' else CIFAR100(train = False, **kwargs)

    return (DataLoader(trainset, batch_size = batch_size, num_workers=num_workers, shuffle=True),
            DataLoader(testset, batch_size = batch_size, num_workers=num_workers, shuffle=False))

def get_cifar10_dataloader(root, batch_size, num_workers):
    return get_cifar_dataloader('cifar10', root, batch_size, num_workers)

def get_cifar100_dataloader(root, batch_size, num_workers):
    return get_cifar_dataloader('cifar100', root, batch_size, num_workers)
