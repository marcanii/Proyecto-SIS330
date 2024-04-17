import torch
from Agent import Agent

if __name__ == '__main__':
    agent = Agent(5, 3*60*108)
    observation = torch.randn(1, 3, 60, 108)
    action, probs, value = agent.choose_action(observation)
    print("Action: ", action)
    print("Probs: ", probs) # probs se utiliza para calcular el ratio entre las nuevas y antiguas políticas
    print("Value: ", value)