from torch import nn
from torchscope import scope
from torchvision import models


class GazeEstimationModel(nn.Module):
    def __init__(self):
        super(GazeEstimationModel, self).__init__()
        resnet = models.resnet50(pretrained=True)
        # Remove linear layer
        modules = list(resnet.children())[:-1]
        self.features = nn.Sequential(*modules)
        # building last several layers
        self.fc = nn.Linear(2048, 3)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = GazeEstimationModel()
    scope(model, input_size=(3, 224, 224))
