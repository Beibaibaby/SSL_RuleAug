import argparse
import os
import torch


parser = argparse.ArgumentParser(description='PyTorch Abstract Reasoning')

# dataset settings
parser.add_argument('--dataset-name', default='RAVEN-FAIR', type=str,
                    help='dataset name')
parser.add_argument('--few-shot-p', default=1., type=float,
                    help='how many percentage of data will be used for training (for few-shot learning)')
parser.add_argument('--image-size', default=80, type=int,
                    help='image size')
# network settings
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture (default: resnet18)')
parser.add_argument('--num-extra-stages', default=1, type=int,
                    help='number of extra normal residue blocks or predictive coding blocks')
parser.add_argument('--block-drop', default=0.0, type=float,
                    help="dropout within each block")
parser.add_argument('--classifier-drop', default=0.0, type=float,
                    help="dropout within classifier block")


# training settings
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--optim', default="SGD", type=str,
                    help='optimizer')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--loss-type', default='ce', 
                    help='choice for losses')
parser.add_argument('--early-stopping', default=20, type=int,
                    help="early stopping for training")
parser.add_argument('--exps', type=str,
                    help='experiment times')

# others settings
parser.add_argument("--ckpt", default="./ckpts/", 
                    help="folder to output checkpoints")
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')


def parse_list(list, cast_type=int):
    list = list.split(',')
    list_values = []
    for l in list:
        list_values.append(cast_type(l))
    return list_values


def main():
    args = parser.parse_args()
    trials = args.exps.split(',')
    all_accs = []
    for trial in trials:
        args.seed = trial
        acc = print_each(os.path.join(args.ckpt, "exp_" + trial + "/"), args)
        all_accs.append(acc)
    all_accs = torch.tensor(all_accs)
    mean_acc = all_accs.mean()
    std_acc = all_accs.std()

    for i, acc in enumerate(all_accs):
        print("{} - {} | exp {} - acc: {:3f}".format(
            args.dataset_name.rjust(10), args.arch, trials[i], acc
    ))

    print("---------------- Overall Performance -----------------")
    print("{} - {} | mean_acc: {:3f} | std_acc: {:.3f}".format(
        args.dataset_name.rjust(10), args.arch, mean_acc, std_acc
    ))
    print(" ")
    


def print_each(fpath, args):

    # args.dataset_dir = os.path.join(args.dataset_dir, args.dataset_name)

    # Parsing for save directory
    fpath += args.dataset_name
    fpath += "-" + args.arch

    if args.block_drop > 0.0 or args.classifier_drop > 0.0:
        fpath += "-b" + str(args.block_drop) + "c" + str(args.classifier_drop)

    fpath += "-" + args.optim
    fpath += "-" + args.loss_type
    fpath += "-imsz" + str(args.image_size)
    fpath += "-wd" + str(args.weight_decay)
    fpath += "-ep" + str(args.epochs)
    fpath += "-es" + str(args.early_stopping)


    if args.seed is not None:
        fpath += '-seed' + str(args.seed)


    fpath = os.path.join(fpath, "log.txt")
    lineinfos = open(fpath).readlines()
    res_infos = lineinfos[-1].strip()
    res_infos = res_infos.split(' ')
    test_accs = res_infos[8]

    # print("{} - {} | fnorm: {} | hnorm: {} | acc: {}".format(
    #     args.dataset_name.rjust(10), args.arch, args.feat_norm_type, args.head_norm_type, test_accs
    # ))

    return float(test_accs)


if __name__ == '__main__':
    main()