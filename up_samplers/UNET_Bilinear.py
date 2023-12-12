from torch import nn


class UpSample(nn.Module):
    def __init__(self, scaling_factor, inp_features, out_features):
        super(UpSample, self).__init__()

        self.up_sample = nn.UpsamplingBilinear2d(scale_factor=scaling_factor)
        print(f'inp features: {inp_features} out features: {out_features}')
        self.conv = nn.Conv2d(inp_features, out_features, kernel_size=3, padding=1)

    def forward(self, x):
        up = self.up_sample(x)
        print(f'up: {up.shape}')
        conv = self.conv(up)
        print(f'conv: {conv.shape}')
        return conv