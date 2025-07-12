import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def get_backbone(name='resnet18', dim=128):
    model = models.__dict__[name](pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, dim)
    return model

class ClusterHead(nn.Module):
    def __init__(self, n_clusters=10, feature_dim=128):
        super().__init__()
        self.linear = nn.Linear(feature_dim, n_clusters)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        return F.softmax(self.linear(x), dim=1)
