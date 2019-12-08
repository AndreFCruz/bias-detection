"""
Several utils functions for storing, updating, and showing average meters.
From: https://github.com/pytorch/examples/blob/master/imagenet/main.py
"""

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_epochs, *meters, prefix=None):
        self.fmtstr = self._get_fmtstr(num_epochs)
        self.meters = meters
        self.prefix = prefix

    def print(self, epoch, **kwargs):
        entries = [self.prefix] if self.prefix is not None else []
        entries += [self.fmtstr.format(epoch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries), **kwargs)

    def _get_fmtstr(self, num_epochs):
        """Format string"""
        num_digits = len(str(num_epochs // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_epochs) + ']'
