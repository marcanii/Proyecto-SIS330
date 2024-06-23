import torchvision.models as models
from ActorNetwork import DenseNetActor169, Efficientnet
import torch

if __name__ == '__main__':
    model = models.densenet121(pretrained=True)
    print(model.features[0])
    print(model.classifier)
    # x = torch.randn(1, 1, 60, 108).to("cuda:0")
    # y = model(x)
    # print(y)