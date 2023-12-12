from torch import nn


class DownSample(nn.Module):
    def __init__(self, scaling_factor):
        super(DownSample, self).__init__()

        self.down_sample = nn.MaxPool2d(kernel_size=scaling_factor, stride=scaling_factor)

    def forward(self, x):
        return self.down_sample(x)
