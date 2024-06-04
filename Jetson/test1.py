from Agent import Agent
import torch

agent = Agent()
if agent.loadModels():
    print("Models loaded successfully")
else:
    print("Error loaded models")
print("FIN")