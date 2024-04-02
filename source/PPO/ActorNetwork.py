import torch.nn as nn
import torchvision.models as models

class ActorNetwork(nn.Module):
    def __init__(self, n_outputs=5, pretrained=False, freeze=False):
        super(ActorNetwork, self).__init__()
        
        resnet50 = models.resnet50(weights=None if not pretrained else 'imagenet')

        self.resnet50 = nn.Sequential(*list(resnet50.children())[:-1])
        if freeze:
            for param in self.resnet50.parameters():
                param.requires_grad=False
        # añadimos una nueva capa lineal para llevar a cabo la clasificación
        self.fc = nn.Linear(2048, n_outputs)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        x = self.resnet50(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.softmax(x)
        return x

    def unfreeze(self):
        for param in self.resnet50.parameters():
            param.requires_grad=True

class ModelCustom(nn.Module):
    def __init__(self, n_outputs=5, pretrained=False, freeze=False):
        super(ModelCustom, self).__init__()
        
        efficientnet = models.efficientnet_v2_s(weights=None if not pretrained else 'imagenet')

        self.efficientnet = nn.Sequential(*list(efficientnet.children())[:-1])
        if freeze:
            for param in self.efficientnet.parameters():
                param.requires_grad=False
        # añadimos una nueva capa lineal para llevar a cabo la clasificación
        self.fc = nn.Linear(1280, n_outputs)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        x = self.efficientnet(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.softmax(x)
        return x

    def unfreeze(self):
        for param in self.efficientnet.parameters():
            param.requires_grad=True