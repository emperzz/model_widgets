from .cifar import get_cifar_dataloader
from .tiny_image_net import get_tiny_imagenet_dataloader

loader_dict = {
    'cifar' : get_cifar_dataloader,
    'tiny-imagenet' : get_tiny_imagenet_dataloader
}