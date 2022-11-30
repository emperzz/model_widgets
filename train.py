def train_epoch(net, optimizer, loss, train_iter, test_iter, num_epochs, device):
    net.train()
    net.to(device)

    for epoch in num_epochs:
        for i, (x,y) in enumerate(train_iter):
            x = x.to(device)
            y = y.to(device)
            y_hat = net(x)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            