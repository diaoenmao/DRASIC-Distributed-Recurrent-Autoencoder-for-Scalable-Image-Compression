from torch.autograd.function import Function


class Quantize(Function):
    def __init__(self):
        super(Quantize, self).__init__()

    @staticmethod
    def forward(ctx, input, is_training):
        x = input.clone()
        if is_training:
            prob = x.new(x.size()).uniform_()
            x[(x + 1) / 2 >= prob] = 1
            x[(x + 1) / 2 < prob] = -1
        else:
            x[(x >= 0) & (x <= 1)] = 1
            x[(x >= -1) & (x < 0)] = -1
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None