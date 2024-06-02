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
        #resnet50 = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1')
        resnet18 = models.resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")
        resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet18 = nn.Sequential(*list(resnet18.children())[:-1])
        if freeze:
            for param in self.resnet18.parameters():
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

# --------------- Modelo Efficientnet ----------------
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

# ---------------- Modelo MobileNet ----------------
class MobileNetActor(nn.Module):
    def __init__(self, n_outputs, alpha, cuda, pretrained=False, freeze=False, chkpt_dir='tmp/ppo'):
        super(MobileNetActor, self).__init__()
        mobilenet = models.mobilenet_v3_small(pretrained=pretrained)
        #mobilenet.features[0][0] = nn.Conv2d(2, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.mobilenet = nn.Sequential(*list(mobilenet.children())[:-1])
        if freeze:
            for param in self.mobilenet.parameters():
                param.requires_grad=False
        # añadimos una nueva capa lineal para llevar a cabo la clasificación
        self.classifier = nn.Sequential(
            nn.Linear(576, 1024),
            nn.Hardswish(),
            nn.Dropout(0.2, inplace=True),
            nn.Linear(1024, n_outputs),
            nn.Softmax(dim=-1)
        )

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_ppo')
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        if cuda:
            print("ActorNetwork esta usando CUDA...")
            self.device = "cuda:0"
        else:
            print("ActorNetwork esta usando CPU...")
            self.device = "cpu"
        self.to(self.device)
        
    def forward(self, x):
        x = self.mobilenet(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        x = Categorical(x)
        return x

    def unfreeze(self):
        for param in self.mobilenet.parameters():
            param.requires_grad=True
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class MyModelActorCNN(nn.Module):
    def __init__(self, n_outputs, alpha, cuda, chkpt_dir='tmp/ppo'):
        super(MyModelActorCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64*4*10, 512)
        self.fc2 = nn.Linear(512, n_outputs)
        self.softmax = nn.Softmax(dim=-1)

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_ppo')
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        if cuda:
            print("ActorNetwork esta usando CUDA...")
            self.device = "cuda:0"
        else:
            print("ActorNetwork esta usando CPU...")
            self.device = "cpu"
        self.to(self.device)
        
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        #print("Conv1: ", x.shape)
        x = nn.functional.relu(self.conv2(x))
        #print("Conv2: ", x.shape)
        x = nn.functional.relu(self.conv3(x))
        #print("Conv3: ", x.shape)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        x = Categorical(x)
        return x

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))