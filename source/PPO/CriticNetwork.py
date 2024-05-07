import torch.nn as nn
import torch
import torch.optim as optim
import os

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, chkpt_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()
        
        self.lineal1 = nn.Linear(input_dims, 256)
        self.lineal2 = nn.Linear(256, 512)
        self.lineal3 = nn.Linear(512, 256)
        self.lineal4 = nn.Linear(256, 1)

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_ppo')
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        # if torch.cuda.is_available():
        #     print("CriticNetwork esta usando CUDA...")
        #     self.device = "cuda:0"
        # else:
        #     print("CriticNetwork esta usando CPU...")
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