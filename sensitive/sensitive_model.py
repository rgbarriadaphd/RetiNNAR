"""
# Author = ruben
# Date: 14/3/24
# Project: RetiNNAR
# File: sensitive_model.py

Description: Implementation of SensitiveNet, in charge of retinal agnostic representation
"""
import torch
import torch.nn as nn

from utils import utils, logger

log = logger.Logger().log()
utils = utils.Utils("config/retinnar.json")
config = utils.get_config()


class SensitiveReguralizer:
    """Adversarial sensitive regularizer.

     Measures the amount of sensitive information present in the learned model represented """

    def __call__(self, *args, **kwargs):
        """
        Λ(𝐱) = log(1 + |0.9 − 𝑃𝑘(𝐷𝑖|𝛗(𝐱𝑖|𝐰∗, 𝐰SN), 𝐰𝑘∗)|)
        """


class TripletLoss:
    """Computes the triplet loss function"""

    def __call__(self, anchor, positive, negative):
        a = torch.flatten(torch.cdist(anchor, negative, p=2)).item()
        b = torch.flatten(torch.cdist(anchor, positive, p=2)).item()
        return a - b


class SensitiveLoss(nn.Module):
    """Computes sensitive net loss"""

    def __init__(self):
        super(SensitiveLoss, self).__init__()
        self._triplet_distance = None
        self._sensitive_regularizer = None

    def forward(self, anchor, positive, negative):
        """"""


class SensitiveDetector(nn.Module):
    """Softmax classification layer in charge of detecting """

    def __init__(self, input_size, output_size):
        super(SensitiveDetector, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.fc(x)
        return torch.softmax(x, dim=1)


class SensitiveNet(nn.Module):
    """"""

    def __init__(self, input_size, num_layers, hidden_size, output_size):
        super(SensitiveNet, self).__init__()
        layers = [nn.Linear(input_size, hidden_size)]

        # Hidden layers
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())  # activation: ReLU

        layers.append(nn.Linear(hidden_size, output_size))  # Output layer
        layers.append(nn.LayerNorm(output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    # random input sample
    input_data = torch.randn(8, (4096 * 3))

    # Init model
    model = SensitiveNet(input_size=(4096 * 3),  # x: input formed by anchor, positive and negative images features
                         num_layers=5,  # Number of sensitive layers
                         hidden_size=1024,  # Units in layers
                         output_size=(4096 * 3),  # output vector size
                         )
    log.info(model)

    # get output
    outputs = model(input_data)

    log.info("Output model size:")
    log.info(outputs.size())
