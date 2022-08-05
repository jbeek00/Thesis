import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn as nn

# One block in the Resnet architecture
class block(nn.Module):
    # Initialization. Takes in and out channels.
    # Id Downsample --> Conv layer in case we change the input size or num. channels
    def __init__(self, in_channels, out_channels, identity_downsample = None, stride = 1):
        super(block, self).__init__()
        # 4: Number of channels after a block is always four times what it was when it entered
        self.expansion = 4
        # First convolution.
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0)
        # Normalize batch between the first and next convolution (batch norm).
        self.bn1 = nn.BatchNorm2d(out_channels)
        # Repeat upper two lines for next convolution (alter parameters)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size = 1, stride = 1, padding = 0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion) # a.k.a. out_channels*4
        # Define activation function (ReLU Layer)
        self.relu = nn.ReLU()
        # conv layer that we do to the identity mapping to normalize shape later on in the layers
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        # Some computations. Initialize x with all things mentioned in the initialization
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        # Run it through the identity_downsample layer from initialization if we need to change shape in some way
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)


        x += identity
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    # Block = block class.
    # Layers = list, how many times we want to use the block. Resnet50 -> [3 first layer, 4 second layer, 6, 3]
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        # output will always be 64 (with 64 input)
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size = 7, stride = 2, padding = 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        # ResNet Layers
        self.layer1 = self._make_layer(block,layers[0], out_channels = 64, stride = 1)
        self.layer2 = self._make_layer(block,layers[1], out_channels = 128, stride = 2)
        self.layer3 = self._make_layer(block,layers[2], out_channels = 256, stride = 2)
        self.layer4 = self._make_layer(block,layers[3], out_channels = 512, stride = 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes) # fc = fully connected layer (of all above, mapped to num classes)

    # Forward pass on layers above
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Send through average pooling to get it in the right shape (so we can send it in the fc layer)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    # Make layer creates layer. Firstly, how many times it uses the block (resid), number of out channels when
    # we're done with the layer, stride (how many more or less times you call the block --> [3, (4-3 --> 1)4, (2)6, 3]
    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        # When do we id downsample (when conv layer changes the identity)?
        # Either unbalanced inp size or stride not 1
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*4, kernel_size = 1,
                                                         stride = stride),
                                               nn.BatchNorm2d(out_channels * 4))

        # This part is the layer that changes the number of channels. For first layer will be 256
        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
        # Output will be 64 (out channels) * 4 (256) at the end of this block
        self.in_channels = out_channels * 4

        # Residual block --> Number of times a block is used
        for i in range(num_residual_blocks - 1):

            layers.append(block(self.in_channels, out_channels)) # 256 --> 64, 64 * 4 --> 256. Stride = 1

        return nn.Sequential(*layers)

# Select resnet architecture
def ResNet50(img_channels = 3, num_classes = 1):
    return ResNet(block, [3, 4, 6, 3], img_channels, num_classes)

def ResNet101(img_channels = 3, num_classes = 1):
    return ResNet(block, [3, 4, 23, 3], img_channels, num_classes)

def ResNet152(img_channels = 3, num_classes = 1):
    return ResNet(block, [3, 8, 36, 3], img_channels, num_classes)


def test():
    net = ResNet50(num_classes=1) # num_classes = 1 --> Regression
    x = torch.squeeze(torch.randn(18851, 3, 224, 224)) # 18851 - no. images, 3 channels (rgb)
    y = net(x).to('cuda')
    print(y.shape)
    print(y)

test()
