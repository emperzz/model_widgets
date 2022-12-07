import torch.nn as nn

def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)

def vgg(conv_arch, num_of_cat):
    conv_blks = []
    in_channels = 3
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全连接层部分
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, num_of_cat))

class VGGLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGGLayer, self).__init__()
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size = 4, padding = 1)
        self.act_layer = nn.ReLU()

    def forward(self, x):
        return self.act_layer(self.conv_layer(x))

class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, act = nn.ReLU(), dropout = 0.5):
        super(LinearLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.act = act
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.act(self.linear(x)))

def vgg_layer(num_layers, in_channels, out_channels):
    layers = []
    for _ in range(num_layers):
        layers.append(VGGLayer(in_channels, out_channels))
        in_channels = out_channels
    return layers

class VGGBlock(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels):
        super(VGGBlock, self).__init__()
        self.vgg_layers = nn.ModuleList(
            vgg_layer(num_layers, in_channels, out_channels)
        )
        self.pool_layer = nn.MaxPool2d(kernel_size=2,stride=2)
        
    def forward(self, x):
        for vgg_layer in self.vgg_layers:
            x = vgg_layer(x)
        
        return self.pool_layer(x)

class VGG(nn.Module):
    def __init__(self, init_channels, conv_arch, num_of_cat):
        # super().__init__()
        super(VGG, self).__init__()
        vgg_blocks = []
        in_channels = init_channels
        for (num_convs, out_channels) in conv_arch:
            vgg_blocks.append(vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels
        
        self.vgg_blocks = nn.ModuleList(
            vgg_blocks
        )
        self.flatten = nn.Flatten()
        self.linear_layer1 = LinearLayer(out_channels * 7 * 7, 4096)
        self.linear_layer2 = LinearLayer(4096, 4096)
        self.linear = nn.Linear(4096, num_of_cat)

    def forward(self, x):
        for vgg_block in self.vgg_blocks:
            x = vgg_block(x)
        x = self.flatten(x)
        return self.linear(self.linear_layer2(self.linear_layer1(x)))

if __name__ == '__main__':
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    net = VGG(3, conv_arch, 200)
    
