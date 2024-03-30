import torch

class MaxPooling(torch.nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()
        self.pool1 = torch.nn.MaxPool2d(2, stride=2)
        self.pool2 = torch.nn.MaxPool2d(2, stride=2)
        self.pool3 = torch.nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        x = self.pool1(x)
        x = self.pool2(x)
        x = self.pool3(x)
        return x

class ConvolutionalMaxPooling(torch.nn.Module):
    def __init__(self):
        super(ConvolutionalMaxPooling, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x