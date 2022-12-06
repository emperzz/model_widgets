def train_epoch(net, optimizer, loss, train_iter, test_iter, num_epochs, device):
    net.train()
    net.to(device)

    for epoch in range(num_epochs):
        for i, (x,y) in enumerate(train_iter):
            x = x.to(device)
            y = y.to(device)
            y_hat = net(x)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            if i % 100 ==0:
                  print(f'loss {l:.3f}, step: {i:.3f}, '
          f'epoch: {epoch:.3f}')
            

if __name__ == '__main__':
    from loader.tiny_image_net import get_tiny_imagenet_dataloader
    from models.vgg import vgg
    import torch
    
    net = vgg(((1, 64//4), (1, 128//4), (2, 256//4), (2, 512//4), (2, 512//4)),
               200)
    train_iter, text_iter = get_tiny_imagenet_dataloader('E:/Data/tiny-imagenet-200', batch_size = 128)
    optimizer = torch.optim.SGD(net.parameters(), lr = 0.05)
    loss = torch.nn.CrossEntropyLoss()

    train_epoch(net, optimizer, loss, train_iter, text_iter, 10, 'cuda')