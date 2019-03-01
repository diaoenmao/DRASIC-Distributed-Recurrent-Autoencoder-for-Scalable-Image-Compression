from functions import pixel_unshuffle


class PixelUnShuffle(Module):

    def __init__(self, downscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input):
        return pixel_unshuffle(input, self.downscale_factor)

    def extra_repr(self):
        return 'downscale_factor={}'.format(self.downscale_factor)