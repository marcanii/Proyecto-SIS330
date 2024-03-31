from  ActorNetwork import ActorNetwork
import torch
import torchvision.models as models


if __name__ == '__main__':
    time = 0
    actor = ActorNetwork().to('cuda')
    outputs = actor(torch.randn(4, 3, 59, 105).to('cuda'))
    print(outputs.shape)
    print(outputs)