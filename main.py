import argparse
import torch
import os
import warnings

from models.vgg import VGG
from loader import loader_dict

parser = argparse.ArgumentParser(description='DNN Experimental Benchmarks')

optim_choices = [name for name in torch.optim.__dict__.keys()
                 if callable(torch.optim.__dict__[name]) and not name.startswith('__')]

# visualdl configs
parser.add_argument('--log', default = './log')
parser.add_argument('--host', default = '127.0.0.1')
parser.add_argument('--port', default = 8040, type = int)
parser.add_argument('-n', '--name', default = None)



# train configs
parser.add_argument('-e', '--epochs', default = 10, type = int)
parser.add_argument('-g', '--gpu', default = False, action='store_true')
parser.add_argument('-b', '--batch-size', default = 256, type = int)
parser.add_argument('-l', '--lr', default = 0.03, type = float)
parser.add_argument('--init-weight', default = None, type = str)
parser.add_argument('-p', '--print-freq', default=100, type = int)
parser.add_argument('--optim', default='SGD', type = str)
parser.add_argument('--loss', default = 'CrossEntropyLoss', type = str)
parser.add_argument('--num-workers', default = 10, type = int)

# data
parser.add_argument('--data-name')
parser.add_argument('--data-path')
parser.add_argument('--model')



def main():
    args = parser.parse_args()

    if args.gpu:
        if torch.cuda.is_available():
            args.device = 'cuda'
        else:
            warnings.warn('current running system does not support cuda, will use cpu instead')
            args.device = 'cpu'

    args.log = os.path.join(args.log, args.data_name) #针对数据训练各种模型，检查每个模型的效果


    args.data_name.lower() in ['cifar10', 'cifar100', 'fashion-mnist', 'tiny-imagenet']

    if args.data_name in ['cifar10', 'cifar100']:
        train_iter, test_iter = loader_dict['cifar'](args.data_name, args.data_path, args.batch_size, args.num_workers)
    else:
        train_iter, test_iter = loader_dict[args.data_name](args.data_path, args.batch_size, args.num_workers)

    optimizer = torch.optim.__dict__[args.optim]
    loss = torch.nn.__dict__[args.loss]

    print(args)

if __name__ == '__main__':
    main()