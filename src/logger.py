from collections import Counter
from torch.utils.tensorboard import SummaryWriter


class Logger():
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)
        self.counter = Counter()
        self.tracker = {}

    def append_scalar(self, result, tag):
        for k in result:
            name = '{}/{}'.format(tag, k)
            self.counter[name] += 1
            self.tracker[name] = result[k]
            self.writer.add_scalar(name, result[k], self.counter[name])
        return

    def append_text(self, result, tag):
        for k in result:
            name = '{}/{}'.format(tag, k)
            self.counter[name] += 1
            self.tracker[name] = result[k]
            self.writer.add_text(name, result[k], self.counter[name])
        return

    def close(self):
        self.writer.close()
