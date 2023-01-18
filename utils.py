import torch
import numpy as np
import random

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
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def get_message(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        return ('\t').join(entries)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def binary_accuracy(output, target):
    correct = torch.eq(torch.round(output).type(target.type()), target).view(-1)
    res = torch.sum(correct) / (target.size(0) * target.size(1))

    return res.view(1,1)

def parse_gpus(gpu_ids):
    gpus = gpu_ids.split(',')
    gpu_ids = []
    for g in gpus:
        g_int = int(g)
        if g_int >= 0:
            gpu_ids.append(g_int)
    if not gpu_ids:
        return None
    return gpu_ids
    

def normalize_image(images):
    return (images / 255.0 - 0.5) * 2


def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(rampup_length, current):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

def get_current_consistency_weight(epoch, args, rampup_type="sigmoid"):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    if rampup_type == "sigmoid":
        return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)
    elif rampup_type == "linear":
        return args.consistency * linear_rampup(epoch, args.consistency_rampup)


def ema_update_teacher(model, global_step, ema_decay):
    alpha = min(1 - 1 / (global_step + 1), ema_decay)
    for param_t, param_s in zip(model.module.teacher.parameters(), model.module.student.parameters()):
        param_t.data.mul_(alpha).add_(1 - alpha, param_s.data)


def mixup(inputs, targets, alpha):

    beta = np.random.beta(alpha, alpha)
    beta = max(beta, 1 - beta)

    perms = torch.randperm(inputs.size(0))

    inputs_perm = inputs[perms]
    target_perm = targets[perms]

    inputs_mixed = beta * inputs + (1 - beta) * inputs_perm
    target_mixed = beta * targets + (1 - beta) * target_perm

    return inputs_mixed, target_mixed

        
# seed the sampling process for reproducibility
# https://pytorch.org/docs/stable/notes/randomness.html
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)