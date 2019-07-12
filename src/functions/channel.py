import torch

def channel(input, mode, snr):
    if(mode=='awgn'):
        sigma = 10**(-snr/20)
        noise = sigma*torch.randn(input.size(),device=input.device)
        channel_output = input + noise
    return channel_output