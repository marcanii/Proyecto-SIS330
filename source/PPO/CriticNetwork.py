import torch.nn as nn

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256):
        super(CriticNetwork, self).__init__()

        
        self.lineal1 = nn.Linear(*input_dims, fc1_dims),
        self.relu1 = nn.ReLU(),
        self.lineal2 = nn.Linear(fc1_dims, fc2_dims),
        self.relu2 = nn.ReLU(),
        self.lineal3 = nn.Linear(fc2_dims, 1)
        

    def forward(self, state):
        value = self.lineal1(state)
        value = self.relu1(value)
        value = self.lineal2(value)
        value = self.relu2(value)
        value = self.lineal3(value)
        return value