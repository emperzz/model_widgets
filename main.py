import argparse
import torch
import time
import os
import warnings

from models.vgg import VGG
from loader import loader_dict
from visualdl import LogWriter

parser = argparse.ArgumentParser(description='DNN Experimental Benchmarks')

optim_choices = [name for name in torch.optim.__dict__.keys()
                 if callable(torch.optim.__dict__[name]) and not name.startswith('__')]

# visualdl configs
parser.add_argument('--log', default = '../log')
parser.add_argument('--model-dir')
parser.add_argument('--host', default = '127.0.0.1')
parser.add_argument('--port', default = 8040, type = int)
parser.add_argument('-n', '--name', default = '')



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
parser.add_argument('--data-path', default = '../dataset')
parser.add_argument('--augment', default = False, action = 'store_true')




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

    train_iter, test_iter = loader_dict[args.data_name.lower()](args.data_path, args.batch_size, args.num_workers)
    

    net = VGG((3, 64, 64),
              ((1, 64//4), (1, 128//4), (2, 256//4), (2, 512//4), (2, 512//4)),
              2048,
              10)
    loss = torch.nn.__dict__[args.loss]()
    optimizer = torch.optim.__dict__[args.optim](net.parameters(), lr = args.lr)

    from train import train_epoch

    with LogWriter(logdir = os.path.join(args.log, net.__class__.__name__ + str(int(time.time()))),
                   display_name = args.name) as  writer:
        writer.add_hparams(hparams_dict={'lr' : args.lr, 'batch_size' : args.batch_size, 'optim' : args.optim},
                           metrics_list= ['train/loss', 'train/acc'])
        train_epoch(net, optimizer, loss, train_iter, test_iter, args.epochs, writer)


if __name__ == '__main__':
    main()