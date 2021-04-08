import torch.nn as nn
from torchvision import models


def get_cnn(pretrained_cnn_weights=None, freeze_weights=False,
            default_pretrained=False):

    cnn = models.resnet152(pretrained=default_pretrained)
    n_features = cnn.fc.in_features
    if pretrained_cnn_weights:
        cnn.fc = nn.Linear(n_features,
                           pretrained_cnn_weights['fc.weight'].size(0))
        cnn.load_state_dict(pretrained_cnn_weights)
    cnn.fc = nn.Identity()
    if freeze_weights:
        for p in cnn.parameters():
            p.requires_grad = False

    return cnn, n_features
