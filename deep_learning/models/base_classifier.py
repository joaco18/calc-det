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
            n_inputs = self.model.classifier[1].in_features

        if fc_dims is not None:
            layers_list = []
            for i in range(len(fc_dims)):
                in_neurons = n_inputs if i == 0 else fc_dims[i-1]
                layers_list = layers_list + [
                    (f'fc{i+1}', nn.Linear(in_neurons, fc_dims[i])),
                    (f'act{i+1}', activation),
                    (f'do{i+1}', nn.Dropout(dropout))]
            layers_list.append((f'fc{i+2}', nn.Linear(fc_dims[i], 1)))
            classifier = nn.Sequential(OrderedDict(layers_list))
        else:
            classifier = nn.Sequential(OrderedDict([
                ('do', nn.Dropout(dropout)),
                ('fc', nn.Linear(n_inputs, 1))
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
