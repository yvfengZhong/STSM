class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.min = 1000
        self.avg = 0
        self.max = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.min = val if val < self.min else self.min
        self.max = val if val > self.max else self.max
        self.avg = self.sum / self.count