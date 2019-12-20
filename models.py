import torch
import torch.nn.functional as F
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
        self.fc_look_vec = nn.Linear(2048, 3)
        self.fc_pupil_size = nn.Linear(2048, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        look_vec = self.fc_look_vec(x)
        look_vec = F.normalize(look_vec)
        pupil_size = self.fc_pupil_size(x)
        pupil_size = torch.sigmoid(pupil_size) * 2 - 1.
        return look_vec, pupil_size


class GazeEstimationMobile(nn.Module):
    def __init__(self):
        super(GazeEstimationMobile, self).__init__()
        mobilenet = models.mobilenet_v2(pretrained=True)
        # Remove linear layer
        modules = list(mobilenet.children())[:-1]
        self.features = nn.Sequential(*modules)
        self.pool = nn.AvgPool2d(kernel_size=7)
        # building last several layers
        self.fc_look_vec = nn.Linear(1280, 3)
        self.fc_pupil_size = nn.Linear(1280, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        look_vec = self.fc_look_vec(x)
        look_vec = F.normalize(look_vec)
        pupil_size = self.fc_pupil_size(x)
        pupil_size = torch.sigmoid(pupil_size) * 2 - 1.
        return look_vec, pupil_size


if __name__ == "__main__":
    model = GazeEstimationModel()
    scope(model, input_size=(3, 224, 224))
