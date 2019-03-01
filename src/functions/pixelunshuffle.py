import torch

def pixel_unshuffle(input, downscale_factor):

    batch_size, channels, in_height, in_width = input.size()
    out_channels = channels*(downscale_factor ** 2)
    out_height = in_height // downscale_factor
    out_width = in_width // downscale_factor

    input_view = input.contiguous().view(
        batch_size, channels, out_height, downscale_factor, out_width, downscale_factor)
        
    shuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return shuffle_out.view(batch_size, out_channels, out_height, out_width)