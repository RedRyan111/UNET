from torch import nn


class UpSample(nn.Module):
    def __init__(self, scaling_factor, inp_features, out_features):
        super(UpSample, self).__init__()

        self.up_sample = nn.UpsamplingBilinear2d(scale_factor=scaling_factor)
        self.conv = nn.Conv2d(inp_features, out_features, kernel_size=3, padding=1)

    def forward(self, x):
        up = self.up_sample(x)
        conv = self.conv(up)
        return conv