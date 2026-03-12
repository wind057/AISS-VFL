import torch
import torch.nn as nn

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    

class CNN_bottom(nn.Module):
    def __init__(self, in_channel, img_size, embed_dim):
        super(CNN_bottom,self).__init__()
        self.in_channel = in_channel
        self.img_size = img_size
        act = nn.ReLU
        self.body = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=3, stride=1, padding=1),
            act(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(1)
        )
        self.fc = nn.Linear(self.fc_input_size, embed_dim)

    def forward(self, x):
        out = self.body(x)
        out = self.fc(out)
        return out
    
    @property
    def fc_input_size(self):
        x = torch.randn((1, self.in_channel, self.img_size, self.img_size))
        out = self.body(x)
        return out.shape[1]
    
class ResNet18_bottom(nn.Module):
    def __init__(self, in_channel, img_size, embed_dim):
        super(ResNet18_bottom, self).__init__()
        self.in_channel = in_channel
        self.img_size = img_size
        
        self.body = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            make_layer(ResBlock, 64, 64, 2, stride=1),
            make_layer(ResBlock, 64, 128, 2, stride=2),
            make_layer(ResBlock, 128, 256, 2, stride=2),
            make_layer(ResBlock, 256, 512, 2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1)
        )
        self.fc1 = nn.Linear(self.fc_input_size, embed_dim)

    def forward(self, x):
        out = self.body(x)
        out = self.fc1(out)
        return out
    
    @property
    def fc_input_size(self):
        x = torch.randn((1, self.in_channel, self.img_size, self.img_size))
        out = self.body(x)
        return out.shape[1]

class top(nn.Module):
    def __init__(self, in_channel, class_num):
        super(top, self).__init__()
        self.fc = nn.Linear(in_channel, class_num)
        
    def forward(self, x):
        out = self.fc(x)
        return out

import torch.nn.functional as F
class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
            
    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out

def make_layer(block, in_channels, out_channels, num_blocks, stride):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
        layers.append(block(in_channels, out_channels, stride))
        in_channels = out_channels
    return nn.Sequential(*layers)

        