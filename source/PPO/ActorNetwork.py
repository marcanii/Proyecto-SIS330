import os
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
from torch.distributions.categorical import Categorical

class ActorNetwork(nn.Module):
    def __init__(self, n_outputs, alpha, pretrained=False, freeze=False, chkpt_dir='tmp/ppo'):
        super(ActorNetwork, self).__init__()
        # Reemplaza la primera capa convolucional de ResNet con la capa personalizada
        resnet50 = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
        resnet50.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet50 = nn.Sequential(*list(resnet50.children())[:-1])
        if freeze:
            for param in self.resnet50.parameters():
                param.requires_grad=False
        # añadimos una nueva capa lineal para llevar a cabo la clasificación
        self.fc = nn.Linear(512, n_outputs)
        self.softmax = nn.Softmax(dim=-1)

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_ppo')
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        if torch.cuda.is_available():
            print("ActorNetwork esta usando CUDA...")
            self.device = "cuda:0"
        else:
            print("ActorNetwork esta usando CPU...")
            self.device = "cpu"
        self.to(self.device)
        
    def forward(self, x):
        x = self.resnet50(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.softmax(x)
        x = Categorical(x)
        return x

    def unfreeze(self):
        for param in self.resnet50.parameters():
            param.requires_grad=True
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))





class Efficientnet(nn.Module):
    def __init__(self, n_outputs, alpha, pretrained=False, freeze=False, chkpt_dir='tmp/ppo'):
        super(Efficientnet, self).__init__()
        
        efficientnet = models.efficientnet_b0(weights=None if not pretrained else 'EfficientNet_B0_Weights.IMAGENET1K_V1')

        self.efficientnet = nn.Sequential(*list(efficientnet.children())[:-1])
        if freeze:
            for param in self.efficientnet.parameters():
                param.requires_grad=False
        # añadimos una nueva capa lineal para llevar a cabo la clasificación
        self.classifier = nn.Sequential(
            nn.Dropout(0.2, inplace=True),
            nn.Linear(1280, n_outputs),
            nn.Softmax(dim=-1)
        )

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_ppo')
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        if torch.cuda.is_available():
            print("CriticNetwork esta usando CUDA...")
            self.device = "cuda:0"
        else:
            print("CriticNetwork esta usando CPU...")
            self.device = "cpu"
        self.to(self.device)
        
    def forward(self, x):
        x = self.efficientnet(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        x = Categorical(x)
        return x

    def unfreeze(self):
        for param in self.efficientnet.parameters():
            param.requires_grad=True
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))