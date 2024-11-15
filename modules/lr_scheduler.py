import math


class CosineScheduler:
    def __init__(self, max_epochs, base_lr=0.01, final_lr=0, warmup_epochs=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_epochs = max_epochs
        self.final_lr = final_lr
        self.warmup_epochs = warmup_epochs
        self.warmup_begin_lr = warmup_begin_lr
        self.max_epochs = self.max_epochs - self.warmup_epochs

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) \
                       * float(epoch) / float(self.warmup_epochs)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_epochs:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_epochs:
            self.base_lr = self.final_lr + (
                self.base_lr_orig - self.final_lr) * (1 + math.cos(
                math.pi * (epoch - self.warmup_epochs) / self.max_epochs)) / 2
        return self.base_lr
