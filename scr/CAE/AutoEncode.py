import torch
import torch.nn as nn

class AutoEncode(nn.Module):
    def __init__(self):
        super(AutoEncode, self).__init__()
        ########################
        # Encoder Architecture #
        ########################
        self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1)# input: 3*840*472, output: 64*420*236
        self.relu1 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(64, 16, 3, stride=2, padding=1)# input: 64*420*236, output: 16*210*118
        self.relu2 = nn.ReLU(True)
        ########################
        # Decoder Architecture #
        ########################
        self.deconv1 = nn.ConvTranspose2d(16, 64, 3, stride=2, padding=1, output_padding=1) # input: 16*210*118, output: 64*420*236
        self.relu3 = nn.ReLU(True)
        self.deconv2 = nn.ConvTranspose2d(64, 3, 3, stride=2, padding=1, output_padding=1)# input: 64*420*236, output: 3*105*59
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        print("conv1: ",x.shape)
        x = self.relu1(x)
        x = self.conv2(x)
        print("conv2: ",x.shape)
        x = self.deconv1(x)
        print("deconv1: ",x.shape)
        x = self.relu3(x)
        x = self.deconv2(x)
        print("deconv2: ",x.shape)
        x = self.sigmoid(x)
        return x
