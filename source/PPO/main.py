from Agent import Agent
import torch

if __name__ == '__main__':
    agente = Agent(5, 3*59*105)
    observation = torch.randn(4, 3*59*105)
    action, probs, value = agente.choose_action(observation)
    print(action, probs, value)