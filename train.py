from visualdl import LogWriter
from visualdl.server import app
from utils import evaluate_accuracy
import torch
import os

def init_visualdl(logdir, model = None, net = None, dummy_input = None):
    if model:
        #检查模型文件是否存在
        if not os.path.exists(model):
            # 如果不存在，且输入了net，则必须输入dummy_input用以生成新的onnx文件
            if net:
                assert isinstance(net, torch.nn.Module)
                assert dummy_input is not None
                torch.onnx.export(net, dummy_input, model)
    
    app.run(logdir,
            model=model,
            host="127.0.0.1",
            port=8040,
            cache_timeout=20,
            language=None,
            public_path=None,
            api_only=False,
            open_browser=False)

def train_epoch(net, optimizer, loss, train_iter, test_iter, num_epochs, logdir = None):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if logdir:
        dummy_input = next(iter(train_iter))[0]
        init_visualdl(logdir, os.path.join(logdir,'model.onnx'), net, dummy_input)


    net.train() 
    net.to(device)
    
    n = 0
    num_batches = len(train_iter)
    with LogWriter(logdir = logdir) as writer:
        for epoch in range(num_epochs):
            for x, (y, boxes) in train_iter:
                x = x.to(device)
                y = y.to(device)
                y_hat = net(x)
                l = loss(y_hat, y)
                l.backward()
                optimizer.step()
                with torch.no_grad():
                    acc = evaluate_accuracy(y, y_hat)
                if logdir:
                    writer.add_scalar(tag = 'train/loss', value = l, step = n )
                    writer.add_histogram(tag = 'train/hist', values = net.linear.weight.data.flatten(), step = n)
                elif n % 100 ==0:
                    print(f'loss {l:.3f}, acc: {acc}, step: {n}, 'f'epoch: {epoch}')
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
    from models.vgg import VGG
    import torch
    
    net = VGG((3, 64, 64),
              ((1, 64//4), (1, 128//4), (2, 256//4), (2, 512//4), (2, 512//4)),
              2048,
              200)
    train_iter, text_iter = get_tiny_imagenet_dataloader('/Users/yangxi/Downloads/tiny-imagenet-200', batch_size = 128)
    optimizer = torch.optim.Adam(net.parameters(), lr = 0.01)
    loss = torch.nn.CrossEntropyLoss()

    train_epoch(net, optimizer, loss, train_iter, text_iter, 10,None)#'/Users/yangxi/Project/log/model_widgets/test')