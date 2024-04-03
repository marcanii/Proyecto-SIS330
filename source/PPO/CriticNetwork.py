import torch.nn as nn

class CriticNetwork(nn.Module):
    def __init__(self, input_dims):
        super(CriticNetwork, self).__init__()

        self.lineal1 = nn.Linear(input_dims, 256)
        self.lineal2 = nn.Linear(256, 512)
        self.lineal3 = nn.Linear(512, 256)
        self.lineal4 = nn.Linear(256, 1)
        
    def forward(self, state):
        value = nn.functional.relu(self.lineal1(state))
        value = nn.functional.relu(self.lineal2(value))
        value = nn.functional.relu(self.lineal3(value))
        value = self.lineal4(value)
        return value