from  ActorNetwork import ActorNetwork, ModelCustom
import torch
import torchvision.models as models

if __name__ == '__main__':
    # actor = ActorNetwork()
    # outputs = actor(torch.randn(4, 3, 59, 105))
    # print(outputs.shape)
    # print(outputs)
    #efficientnet = models.efficientnet_v2_s(pretrained=True)
    efficientnet = ModelCustom()
    outputs = efficientnet(torch.randn(4, 3, 59, 105))
    print(outputs.shape)
    print(outputs)