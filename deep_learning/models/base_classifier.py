from pathlib import Path
thispath = Path.cwd().resolve()
import sys; sys.path.insert(0, str(thispath.parent))

import torch.nn as nn
from torchvision import models
from collections import OrderedDict


class CNNClasssifier:
    def __init__(
        self, activation: nn.Module = nn.LeakyReLU(), dropout: float = 0.5,
        fc_dims: tuple = (512, 512), freeze_weights: bool = False,
        backbone: str = 'resnet18', pretrained: bool = True
    ):
        self.model = getattr(models, backbone)
        if pretrained:
            self.model = self.model(pretrained=pretrained)
        else:
            self.model = self.model()

        if freeze_weights:
            for param in self.model.parameters():
                param.requires_grad = False

        if hasattr(self.model, 'fc'):
            n_inputs = self.model.fc.in_features
        else:
            n_inputs = self.model.classifier[0].in_features
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(n_inputs, fc_dims[0])),
            ('act1', activation),
            ('do1', nn.Dropout(dropout)),
            ('fc2', nn.Linear(fc_dims[0], fc_dims[1])),
            ('act2', activation),
            ('do2', nn.Dropout(dropout)),
            ('fc3', nn.Linear(fc_dims[1], 1))
        ]))

        if hasattr(self.model, 'fc'):
            self.model.fc = classifier
        else:
            self.model.classifier = classifier

        self.model.apply(self.initialize_fc_weights)

    @staticmethod
    def initialize_fc_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)
