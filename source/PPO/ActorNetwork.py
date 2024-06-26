import os
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
from torch.distributions.categorical import Categorical

class ActorNetwork(nn.Module):
    def __init__(self, n_outputs, alpha, cuda, freeze=False, chkpt_dir='source/PPO/tmp/ppo/'):
        super(ActorNetwork, self).__init__()
        # Reemplaza la primera capa convolucional de ResNet con la capa personalizada
        resnet18 = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
        #resnet18 = models.resnet152(weights="ResNet152_Weights.IMAGENET1K_V1")
        resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet18 = nn.Sequential(*list(resnet18.children())[:-1])
        if freeze:
            for param in self.resnet18.parameters():
                param.requires_grad=False
        # añadimos una nueva capa lineal para llevar a cabo la clasificación
        self.fc = nn.Linear(512, n_outputs)
        self.softmax = nn.Softmax(dim=-1)

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_ppo_transfer.pt')
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        if cuda:
            print("ActorNetwork esta usando CUDA...")
            self.device = "cuda:0"
        else:
            print("ActorNetwork esta usando CPU...")
            self.device = "cpu"
        self.to(self.device)
        
    def forward(self, x):
        x = self.resnet18(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.softmax(x)
        x = Categorical(x)
        return x

    def unfreeze(self):
        for param in self.resnet18.parameters():
            param.requires_grad=True
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

# ---------------- Modelo DenseNet201 ----------------
class DenseNetActor201(nn.Module):
    def __init__(self, n_outputs, alpha, cuda, freeze=False, chkpt_dir='source/PPO/tmp/ppo/'):
        super(DenseNetActor201, self).__init__()
        
        densenet = models.densenet201(weights='DenseNet201_Weights.IMAGENET1K_V1')
        densenet.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.densenet = nn.Sequential(*list(densenet.children())[:-1])
        if freeze:
            for param in self.densenet.parameters():
                param.requires_grad=False
        # añadimos una nueva capa lineal para llevar a cabo la clasificación
        self.classifier = nn.Sequential(
            nn.Linear(5760, n_outputs),
            nn.Softmax(dim=-1)
        )

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_ppo_densenet_201_v2.pt')
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        if cuda:
            print("ActorNetwork esta usando CUDA...")
            self.device = "cuda:0"
        else:
            print("ActorNetwork esta usando CPU...")
            self.device = "cpu"
        self.to(self.device)
        
    def forward(self, x):
        x = self.densenet(x)
        x = x.view(x.shape[0], -1)
        #print("Salida DenseNet: ",x.shape)
        x = self.classifier(x)
        x = Categorical(x)
        return x

    def unfreeze(self):
        for param in self.densenet.parameters():
            param.requires_grad=True
        print("DenseNet unfreezed")
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

# ---------------- Modelo DenseNet121 ----------------
class DenseNetActor121(nn.Module):
    def __init__(self, n_outputs, alpha, cuda, freeze=False, chkpt_dir='source/PPO/tmp/ppo/'):
        super(DenseNetActor121, self).__init__()
        
        densenet = models.densenet121(weights='DenseNet121_Weights.IMAGENET1K_V1')
        densenet.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.densenet = nn.Sequential(*list(densenet.children())[:-1])
        if freeze:
            for param in self.densenet.parameters():
                param.requires_grad=False
        # añadimos una nueva capa lineal para llevar a cabo la clasificación
        self.classifier = nn.Sequential(
            nn.Linear(3072, n_outputs),
            nn.Softmax(dim=-1)
        )

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_ppo_densenet_121.pt')
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        if cuda:
            print("ActorNetwork esta usando CUDA...")
            self.device = "cuda:0"
        else:
            print("ActorNetwork esta usando CPU...")
            self.device = "cpu"
        self.to(self.device)
        
    def forward(self, x):
        x = self.densenet(x)
        x = x.view(x.shape[0], -1)
        #print("Salida DenseNet: ",x.shape)
        x = self.classifier(x)
        x = Categorical(x)
        return x

    def unfreeze(self):
        for param in self.densenet.parameters():
            param.requires_grad=True
        print("DenseNet unfreezed")
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))