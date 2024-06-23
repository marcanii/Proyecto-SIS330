import torch.nn as nn
import torch
import torch.optim as optim
import os

class CriticNetwork1(nn.Module):
    def __init__(self, input_dims, alpha, cuda, chkpt_dir='tmp/ppo'):
        super(CriticNetwork1, self).__init__()
        
        self.lineal1 = nn.Linear(input_dims, 256)
        self.lineal2 = nn.Linear(256, 512)
        self.lineal3 = nn.Linear(512, 256)
        self.lineal4 = nn.Linear(256, 1)

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_ppo')
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        if cuda:
            print("CriticNetwork esta usando CUDA...")
            self.device = "cuda:0"
        else:
            print("CriticNetwork esta usando CPU...")
            self.device = "cpu"
        self.to(self.device)
        
    def forward(self, state):
        value = nn.functional.relu(self.lineal1(state))
        value = nn.functional.relu(self.lineal2(value))
        value = nn.functional.relu(self.lineal3(value))
        value = self.lineal4(value)
        return value
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, alpha, cuda, chkpt_dir='source/PPO/tmp/ppo/'):
        super(CriticNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64*4*10, 512) # 64*56*104 for (480x864) and 64*4*10 for (60x108)
        self.fc2 = nn.Linear(512, 1)

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_ppo.pt')
        self.optimizer = optim.Adam(self.parameters(), lr=alpha*10)
        if cuda:
            print("CriticNetwork esta usando CUDA...")
            self.device = "cuda:0"
        else:
            print("CriticNetwork esta usando CPU...")
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
        return x

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))