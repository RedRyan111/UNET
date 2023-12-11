from torch import nn


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, scale_factor=2):
        super(unetUp, self).__init__()
        #self.conv = unetConv2(out_size*2, out_size, False)
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)

        # initialise the blocks
        #for m in self.children():
        #    if m.__class__.__name__.find('unetConv2') != -1: continue
        #    init_weights(m, init_type='kaiming')

    def forward(self, x):
        return self.up(x)