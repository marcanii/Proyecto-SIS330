import torch
from PPO.Agent import Agent
import torchvision.models as models

if __name__ == '__main__':
    agent = Agent(5, 2*60*108)
    action, probs, value = agent.choose_action(torch.randn(1, 2, 60, 108))
    print("Action: ", action)
    print("Probs: ", probs)
    print("Value: ", value)