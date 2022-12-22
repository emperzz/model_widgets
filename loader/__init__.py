from .cifar import get_cifar10_dataloader,get_cifar100_dataloader
from .tiny_image_net import get_tiny_imagenet_dataloader
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def image_loader(root, imageclass, batch_size, num_workers, resize = 224):
    
    trans = transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    train_iter = DataLoader(imageclass(root, train = True, transforms = trans) , 
                            shuffle = True, num_workers = num_workers, batch_size=batch_size)
    
    test_iter = DataLoader(imageclass(root, train = False, transforms = trans) , 
                            shuffle = False, num_workers = num_workers, batch_size=batch_size)

    return train_iter, test_iter
    


loader_dict = {
    'cifar10' : get_cifar10_dataloader,
    'cifar100' : get_cifar100_dataloader,
    'tiny-imagenet' : get_tiny_imagenet_dataloader
}