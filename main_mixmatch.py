import argparse
import os
import random
import time
import warnings
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

from data import TwoStreamBatchSampler, ToTensor, TensorFliplr
from utils import AverageMeter, ProgressMeter, accuracy, normalize_image, parse_gpus
from report_acc_regime import init_acc_regime, update_acc_regime
from loss import ContrastLoss, softmax_mse_loss, softmax_kl_loss
from checkpoint import save_checkpoint, load_checkpoint
from thop import profile
from networks import create_model


parser = argparse.ArgumentParser(description='PyTorch Semi-Supervised Learing for Abstract Reasoning')

# dataset settings
parser.add_argument('--dataset-dir', default='datasets/',
                    help='path to dataset')
parser.add_argument('--dataset-name', default='RAVEN-FAIR',
                    help='dataset name')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-ba', '--batch-accum', default=1, type=int,
                    metavar='N',
                    help='number of mini-batches to accumulate gradient over before updating (default: 1)')
parser.add_argument('--image-size', default=256, type=int,
                    help='image size')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--early-stopping', default=0, type=int,
                    help="early stopping for training")


# network settings
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture (default: resnet18)')
parser.add_argument('--block-drop', default=0.0, type=float,
                    help="dropout within each block")
parser.add_argument('--classifier-drop', default=0.0, type=float,
                    help="dropout within classifier block")


# training settings
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--optim', default="adam", type=str,
                    help='optimizer')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate-on-test', action='store_true',
                    help='whether to do evaluation on test set')
parser.add_argument('--loss-type', default='ce', 
                    help='choice for losses')
parser.add_argument('--num-train-samples', default=None, type=int,
                    help='number of training samples')
parser.add_argument('--ignore-index', default=-100, type=int,
                    help='ignore index for unlabeled data')
parser.add_argument('--consistency', default=10, type=float,
                    help='consistency loss weight')
parser.add_argument('--consistency-rampup', default=10, type=float,
                    help='consistency rampup')
parser.add_argument('--consistency_type', default='mse', type=str,
                    help='which type of consistency loss should be used')
parser.add_argument('--ema_decay', default=0.999, type=float,
                    help='ema decay')
parser.add_argument('--K', default=2, type=int,
                    help='augment numbers')
parser.add_argument('--T', default=0.5, type=int,
                    help='temperature')
parser.add_argument('--alpha', default=0.75, type=int,
                    help='mixup alpha')
parser.add_argument('--max-iterations', default=None, type=int,
                    help='max iterations per epoch')


# others settings
parser.add_argument("--ckpt", default="./ckpts/", 
                    help="folder to output checkpoints")
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default="0",
                    help='GPU id to use.')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument("--fp16", action='store_true',
                    help="whether to use fp16 for training")
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on test set')
parser.add_argument('--show-detail', action='store_true',
                    help="whether to show detail accuracy on all sub-types")


def normalize_image(images):
    return (images / 255.0 - 0.5) * 2


def mixup(inputs, targets, alpha):

    beta = np.random.beta(alpha, alpha)
    beta = max(beta, 1 - beta)

    perms = torch.randperm(inputs.size(0))

    inputs_perm = inputs[perms]
    target_perm = targets[perms]

    inputs_mixed = beta * inputs + (1 - beta) * inputs_perm
    target_mixed = beta * targets + (1 - beta) * target_perm

    return inputs_mixed, target_mixed


def augment_data(inputs, augmentor):
    B, C, H, W = inputs.size()
    outputs = [augmentor(ii.reshape(C, 1, H, W)).reshape(1, C, H, W) for ii in inputs]
    return torch.cat(outputs, dim=0)


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def sigmoid_rampup(consistency_rampup, current):
    if consistency_rampup == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, consistency_rampup)
        phase = 1.0 - current / consistency_rampup
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(rampup_length, current):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)
    
def get_current_consistency_weight(consistency, consistency_rampup, epoch):
# Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * linear_rampup(consistency_rampup, epoch)



class WeightEMA(object):
    def __init__(self, model_student, model_teacher, alpha=0.999, lr=0.001):
        self.model_student = model_student
        self.model_teacher = model_teacher
        self.alpha = alpha
        self.params_student = list(model_student.state_dict().values())
        self.params_teacher = list(model_teacher.state_dict().values())
        self.wd = 0.02 * lr

        for param_s, param_t in zip(self.params_student, self.params_teacher):
            param_s.data.copy_(param_t.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param_s, param_t in zip(self.params_student, self.params_teacher):
            
            if param_t.dtype == torch.float32:
                param_t.mul_(self.alpha)
                param_t.add_(param_s * one_minus_alpha)
                # customized weight decay
                param_s.mul_(1 - self.wd)
    
    
class TrainContrastLoss(object):
    def __call__(self, outputs, target):
        loss = -torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * target, dim=1))
        return loss

# seed the sampling process for reproducibility
# https://pytorch.org/docs/stable/notes/randomness.html
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_data_loader(args, data_split = 'train', transform = None, num_train_samples=None, ssl=False):

    if 'RAVEN' in args.dataset_name:
        from data import RAVEN as create_dataset
    elif 'PGM' in args.dataset_name:
        from data import PGM as create_dataset

    dataset = create_dataset(
        args.dataset_dir, 
        data_split = data_split, 
        image_size = args.image_size, 
        transform = transform,
        num_samples = num_train_samples,
        ssl=ssl
    )
    
    if data_split == "train":
        sampler = TwoStreamBatchSampler(
            primary_indices=dataset.unlabeled_indices, 
            secondary_indices=dataset.labeled_indices, 
            primary_batch_size=args.batch_size, 
            secondary_batch_size=args.batch_size//2
        )
        
        if args.seed is not None:
            g = torch.Generator()
            g.manual_seed(args.seed)

            data_loader = torch.utils.data.DataLoader(
                dataset, 
                num_workers=args.workers, 
                pin_memory=False, 
                batch_sampler=sampler,
                generator=g, 
                worker_init_fn=seed_worker, 
                persistent_workers=True
            )
        else:
            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size, shuffle=(data_split == "train"),
                num_workers=args.workers, pin_memory=False, sampler=sampler,
                persistent_workers=True
            )
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=False, sampler=None,
            persistent_workers=True
        )

    return data_loader


best_acc1 = 0
global_step = 0

def main():
    
    global global_step
    global best_acc1
    
    args = parser.parse_args()

    args.dataset_dir = os.path.join(args.dataset_dir, args.dataset_name)

    args.ckpt += args.dataset_name
    args.ckpt += "-" + args.arch

    if args.block_drop > 0.0 or args.classifier_drop > 0.0:
        args.ckpt += "-b" + str(args.block_drop) + "c" + str(args.classifier_drop)

    args.ckpt += "-" + args.optim
    args.ckpt += "-" + args.loss_type
    args.ckpt += "-imsz" + str(args.image_size)
    args.ckpt += "-wd" + str(args.weight_decay)
    args.ckpt += "-ep" + str(args.epochs)
    args.ckpt += "-es" + str(args.early_stopping)

    args.gpu = parse_gpus(args.gpu)
    if args.gpu is not None:
        args.device = torch.device("cuda:{}".format(args.gpu[0]))
    else:
        args.device = torch.device("cpu")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        args.ckpt += '-seed' + str(args.seed)

    else:
        cudnn.deterministic = False
        cudnn.benchmark = True

    if not os.path.isdir(args.ckpt):
        os.makedirs(args.ckpt)

    main_worker(args)


def main_worker(args):
    global best_acc1

    # create model
    model = create_model(args)

    model_student = model.student
    model_teacher = model.teacher
    for param_s, param_t in zip(model_student.parameters(), model_teacher.parameters()):
        param_t.data.copy_(param_s.data)  # initialize
        param_t.detach_()
        param_t.requires_grad = False  # not update by gradient

    log_path = os.path.join(args.ckpt, "log.txt")

    if os.path.exists(log_path):
        log_file = open(log_path, mode="a")
    else:
        log_file = open(log_path, mode="w")
    
    for key, value in vars(args).items():
        log_file.write('{0}: {1}\n'.format(key, value))

    args.log_file = log_file


    model_flops = copy.deepcopy(model_student)
    x = torch.randn(2, 16, args.image_size, args.image_size)
    flops, params = profile(model_flops, inputs=(x,))
    del model_flops

    print("model [%s] - params: %.6fM" % (args.arch, params / 1e6))
    print("model [%s] - FLOPs: %.6fG" % (args.arch, flops / 1e9))
        
    args.log_file.write("--------------------------------------------------\n")
    args.log_file.write("Network - " + args.arch + "\n")
    args.log_file.write("Params - %.6fM" % (params / 1e6) + "\n")
    args.log_file.write("FLOPs - %.6fG" % (flops / 1e9) + "\n")

    if args.evaluate == False:
        print(model)
    
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        torch.cuda.set_device(args.device)
        model_student = model_student.to(args.gpu[0])
        model_student = torch.nn.DataParallel(model_student, args.gpu)

        model_teacher = model_teacher.to(args.gpu[0])
        model_teacher = torch.nn.DataParallel(model_teacher, args.gpu)


    # define loss function (criterion) and optimizer
    if args.loss_type == "ce":
        class_criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    elif args.loss_type == "ct":
        class_criterion = TrainContrastLoss()
        
    
    if args.consistency_type == 'mse':
        consistency_criterion = softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = softmax_kl_loss
    else:
        assert False, args.consistency_type

    
    if args.optim == "sgd":
        optimizer = torch.optim.SGD(model_student.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    elif args.optim == "adam":
        optimizer = torch.optim.Adam(model_student.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    elif args.optim == "adamw":
        optimizer = torch.optim.AdamW(model_student.parameters(), lr = args.lr, weight_decay = args.weight_decay)

    optimizer_ema = WeightEMA(model_student, model_teacher, args.ema_decay, args.lr)


    if args.resume:
        model, optimizer, best_acc1, start_epoch = load_checkpoint(args, model, optimizer)
        args.start_epoch = start_epoch

    # --------------------------------------------------------------------------------------------------------------
    # Create data loader
    tfs = transforms.Compose([ToTensor()])
    train_loader = get_data_loader(args, data_split='train', transform=tfs, num_train_samples=args.num_train_samples, ssl=True)
    args.batch_size = 256
    valid_loader = get_data_loader(args, data_split='val',   transform=tfs)
    test_loader  = get_data_loader(args, data_split='test',  transform=tfs)
    # --------------------------------------------------------------------------------------------------------------


    args.log_file.write("Number of training samples (unlabeled): %d\n" % len(train_loader.dataset.unlabeled_indices))
    args.log_file.write("Number of training samples (labeled): %d\n" % len(train_loader.dataset.labeled_indices))
    args.log_file.write("Number of validation samples: %d\n" % len(valid_loader.dataset))
    args.log_file.write("Number of testing samples: %d\n" % len(test_loader.dataset))

    args.log_file.write("--------------------------------------------------\n")
    args.log_file.close()


    if args.evaluate:
        if args.evaluate_on_test:
            acc = validate(test_loader, model_student, model_teacher, ContrastLoss(), args, valid_set="Test")
        else:
            acc = validate(valid_loader, model_student, model_teacher, ContrastLoss(), args, valid_set="Valid")
        return

    if args.fp16:
        args.scaler = torch.cuda.amp.GradScaler()

    cont_epoch = 0
    best_epoch = 0
    test_acc2  = 0

    args.consistency_rampup = args.epochs
    args.augmentor = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.RandomVerticalFlip(p=0.5),
    ])

    for epoch in range(args.start_epoch, args.epochs):
        
        args.log_file = open(log_path, mode="a")

        if args.optim == "sgd":
            print("adjust learning rate ...")
            adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model_student, model_teacher, class_criterion, consistency_criterion, optimizer, optimizer_ema, epoch, args)

        # evaluate on validation set
        acc1 = validate(valid_loader, model_student, model_teacher, ContrastLoss(), args, valid_set="Valid")
        

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if args.evaluate_on_test and is_best:
            acc2 = validate(test_loader, model_student, model_teacher, ContrastLoss(), args, valid_set="Test")

        save_checkpoint({
            "epoch": epoch + 1,
            "arch": args.arch,
            "state_dict": model.state_dict(),
            "best_acc": best_acc1,
            "optimizer" : optimizer.state_dict(),
            }, is_best, epoch, save_path=args.ckpt)

        if is_best:
            cont_epoch = 0
            best_epoch = epoch
            test_acc2 = acc2
        else:
            cont_epoch += 1

        epoch_msg = ("----------- Best Acc at [{}]: Valid {:.3f} Test {:.3f} Continuous Epoch {} -----------".format(best_epoch, best_acc1, test_acc2, cont_epoch))
        print(epoch_msg)

        args.log_file.write(epoch_msg + "\n")
        args.log_file.close()
        
        if args.early_stopping > 0 and cont_epoch >= args.early_stopping:
            print("Training is done because no performance is improved on [Valid] set in last {} epochs".format(args.early_stopping))
            break



def train(data_loader, model_student, model_teacher, class_criterion, consistency_criterion, optimizer, optimizer_ema, epoch, args):
    
    global global_step
    
    batch_time = AverageMeter('Time', ':.3f')
    data_time = AverageMeter('Data', ':.3f')
    losses_s = AverageMeter('Loss_s', ':.4f')
    losses_u = AverageMeter('Loss_u', ':.4f')
    
    top1 = AverageMeter('Acc_s', ':.2f')

    progress = ProgressMeter(
        args.max_iterations,
        [batch_time, data_time, losses_s, losses_u, top1],
        prefix = "Epoch: [{}]".format(epoch))

    param_groups = optimizer.param_groups[0]
    curr_lr = param_groups["lr"]

    # switch to train mode
    model_student.train()
    data_loader = iter(data_loader)
    # model_teacher.train()
    # model_ema.train()
    accum_track = 0
    end = time.time()
    for i in range(args.max_iterations):
        # measure data loading time

        images, target = next(data_loader)
        images = normalize_image(images)
        data_time.update(time.time() - end)

        labeled_indices = target.ne(args.ignore_index)

        images_s = images[labeled_indices]
        images_u = images[~labeled_indices]

        if args.gpu is not None:
            images_s = images_s.to(args.device, non_blocking=True)
            images_u = images_u.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)

        with torch.no_grad():
            
            # compute guessed labels of unlabel samples
            outputs = []
            images = []
            with torch.no_grad():
                model_student.eval()
                for j in range(args.K):
                    if args.fp16:
                        with torch.cuda.amp.autocast():
                            images_ = augment_data(images_u, args.augmentor)
                            outs = model_student(images_).unsqueeze(-1)
                            outputs.append(outs)
                            images.append(images_)
                model_student.train()

            outputs = torch.cat(outputs, dim=-1)
            outputs = torch.softmax(outputs, dim=1).mean(dim=-1)
            outputs = outputs ** (1 / args.T)
            target_u = outputs / outputs.sum(dim=1, keepdim=True)
            target_u = target_u.detach()

            images = torch.cat(images, dim=0)


        images_s = augment_data(images_s, args.augmentor)
        target_s = torch.zeros(images_s.size(0), target_u.size(1)).to(args.device, non_blocking=True)
        target_s.scatter_(1, target[labeled_indices].view(-1, 1), 1.0)

        all_images = torch.cat([images_s, images], dim=0)
        all_target = torch.cat([target_s, target_u.repeat(args.K, 1)], dim=0)


        all_images_mixed, all_target_mixed = mixup(all_images, all_target, args.alpha)


        # compute output
        if args.fp16:
            with torch.cuda.amp.autocast():

                output = model_student(all_images_mixed)

                target_s, target_u = all_target_mixed[:images_s.size(0)], all_target_mixed[images_s.size(0):]
                csc_weight = get_current_consistency_weight(args.consistency, args.consistency_rampup, epoch)
                # all_target_mixed[:images_s.size(0)], all_target_mixed[images_s.size(0):]

                if type(output) is tuple:
                    
                    loss_s = [class_criterion(oo[:images_s.size(0)], target_s) for oo in output[1]]
                    loss_s = sum(loss_s) / len(output[1]) + class_criterion(output[0][:images_s.size(0)], target_s)
                    
                    loss_u = [consistency_criterion(oo[images_s.size(0):], target_u, is_target_logits_normalized=True) / target_u.size(0) for oo in output[1]]
                    loss_u = sum(loss_u) / len(output[1]) + consistency_criterion(output[0][images_s.size(0):], target_u, is_target_logits_normalized=True) / target_u.size(0)
                    
                    output_s, output_u = output[0][:images_s.size(0)], output[0][images_s.size(0):]

                else:

                    output_s, output_u = output[:images_s.size(0)], output[images_s.size(0):]
                    
                    loss_s = class_criterion(output_s, target_s)
                    loss_u = consistency_criterion(output_u, target_u, is_target_logits_normalized=True) / target_u.size(0) * csc_weight

                losses_s.update(loss_s.item(), target_s.size(0))
                losses_u.update(loss_u.item(), target_u.size(0))
                
            loss = loss_s + loss_u
            
            args.scaler.scale(loss).backward()

            accum_track += 1
            if(accum_track == args.batch_accum):
                args.scaler.step(optimizer)
                args.scaler.update()
                accum_track = 0
                optimizer.zero_grad()
                optimizer_ema.step()
                  
        else:
            raise NotImplementedError("Only fp16 is supported Currently.")
#             # output = model(images)  
#             # loss = criterion(output, target)
#             # losses.update(loss.item(), images.size(0))
# #             # compute gradient and do SGD step
# #             loss.backward()
# #             optimizer.step()

        # global_step += 1
        # ema_update_teacher(model_student, model_teacher, global_step, args.ema_decay)

        # measure accuracy and record loss
        acc1 = accuracy(output_s, target[labeled_indices])
        top1.update(acc1[0][0], labeled_indices.sum())  

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or i == args.max_iterations - 1:
            epoch_msg = progress.get_message(i)
            epoch_msg += ("\tLr  {:.4f}".format(curr_lr))
            args.log_file.write(epoch_msg + "\n")
            print(epoch_msg)


def validate(data_loader, model_student, model_teacher, criterion, args, valid_set='Valid'):
    

    batch_time = AverageMeter('Time', ':.3f')
    losses_s = AverageMeter('Loss_s', ':.3f')
    losses_t = AverageMeter('Loss_t', ':.3f')

    top1_s = AverageMeter('Acc_s', ':.2f')
    top1_t = AverageMeter('Acc_t', ':.2f')

    progress = ProgressMeter(
        len(data_loader),
        [batch_time, losses_s, losses_t, top1_s, top1_t],
        prefix = valid_set + ': ')

    # switch to evaluate mode
    model_student.eval()
    model_teacher.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target, meta_target, structure_encoded, data_file) in enumerate(data_loader):
           
            if args.gpu is not None:
                images = images.to(args.device, non_blocking=True)
                target = target.to(args.device, non_blocking=True)

            images = normalize_image(images)

            # compute outputs
            output_s = model_student(images)
            output_t = model_teacher(images)
            
            loss = criterion(output_s, target)
            losses_s.update(loss.item(), images.size(0))
            
            loss = criterion(output_t, target)
            losses_t.update(loss.item(), images.size(0))

            # measure accuracy and record loss
            acc1 = accuracy(output_t, target)
            top1_t.update(acc1[0][0], images.size(0))
            
            acc1 = accuracy(output_s, target)
            top1_s.update(acc1[0][0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 or i == len(data_loader) - 1:
                epoch_msg = progress.get_message(i)
                print(epoch_msg)

        # TODO: this should also be done with the ProgressMeter

        epoch_msg = '----------- {valid_set} Acc {mean_acc_s:.3f} | {mean_acc_t:.3f} -----------'.format(
            valid_set=valid_set, mean_acc_s=top1_s.avg, mean_acc_t=top1_t.avg)

        print(epoch_msg)
        
        if args.evaluate == False:
            args.log_file.write(epoch_msg + "\n")


    return top1_t.avg



if __name__ == '__main__':
    main()