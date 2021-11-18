from typing import Any

import torch


class CNN(torch.nn.Module):

    hyperparams: dict[str, Any]

    def __init__(self, hyperparams: dict):
        super().__init__()

        # parameters
        self.hyperparams = hyperparams
        channels = hyperparams["channels"]
        kernels = hyperparams["kernels"]

        # layers
        self.convs = torch.nn.ModuleList([])
        last_ch = 1
        width = 28
        for ch, k in zip(channels, kernels):
            self.convs.append(torch.nn.Conv2d(last_ch, ch, kernel_size=k, stride=1))
            width = (width - k + 1) // 2
            last_ch = ch
        self.out = torch.nn.Linear(ch * width * width, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        for conv in self.convs:
            x = conv(x)
            x = torch.nn.functional.relu(x)
            x = torch.nn.functional.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return torch.nn.functional.softmax(x, dim=-1)
