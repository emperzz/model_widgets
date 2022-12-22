from utils import evaluate_accuracy
import torch
import os



def train_epoch(net, optimizer, loss, train_iter, test_iter, num_epochs, writer = None):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # if logdir:
    #     dummy_input = next(iter(train_iter))[0]
    #     init_visualdl(logdir, os.path.join(logdir,'model.onnx'), net, dummy_input)

    def init_weights(m):
        if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)

    net.to(device)
    net.train() 
    
    n = 0
    num_batches = len(train_iter)
    for epoch in range(num_epochs):
        for x, y in train_iter:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            y_hat = net(x)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                acc = evaluate_accuracy(y, y_hat)
            if writer:
                writer.add_scalar(tag = 'train/loss', value = l.item(), step = n )
                writer.add_scalar(tag = 'train/acc', value = acc, step = n)
            
            if n % 100 ==0:
                print(f'loss {l:.3f}, acc: {acc:.3f}, step: {n}, 'f'epoch: {epoch}')
            n += 1
            

class TrainPipe:
    def __init__(self, net, optimizer, loss_func, 
                 train_iter, test_iter, epochs,
                 log_dir = None):
        self.net = net
        self.optimizer = optimizer
        self.loss = loss_func
        self.log_dir = log_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.epochs = epochs





class DataPipe:
    def __init__(self):
        pass


if __name__ == '__main__':
    from loader.tiny_image_net import get_tiny_imagenet_dataloader
    from loader.cifar import get_cifar_dataloader
    from models.vgg import VGG
    import torch
    import d2l.torch as d2l
    
    net = VGG((3, 64, 64),
              ((1, 64//4), (1, 128//4), (2, 256//4), (2, 512//4), (2, 512//4)),
              2048,
              200)
    train_iter, test_iter = get_tiny_imagenet_dataloader('/Users/yangxi/Downloads/tiny-imagenet-200', batch_size = 256, num_workers=10)
    # train_iter, test_iter = get_cifar_dataloader('cifar10', '/Users/yangxi/Project/dataset', batch_size = 256, num_workers=10)
    optimizer = torch.optim.SGD(net.parameters(), lr = 0.05)
    loss = torch.nn.CrossEntropyLoss()
    # d2l.train_ch6(net, train_iter, test_iter, 10, 0.05, 'cpu')
    train_epoch(net, optimizer, loss, train_iter, test_iter, 20,'/Users/yangxi/Project/log/model_widgets/test')