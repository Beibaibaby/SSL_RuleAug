import os
import torch
import shutil

def save_checkpoint(state, is_best, epoch, save_path='./'):
    print("=> saving checkpoint '{}'".format(epoch))
    torch.save(state, os.path.join(save_path, 'checkpoint.pth.tar'))
    # if(epoch % 10 == 0):
        # torch.save(state, os.path.join(save_path, 'checkpoint_%03d.pth.tar' % epoch))
    if is_best:
        shutil.copyfile(os.path.join(save_path, 'checkpoint.pth.tar'), 
                        os.path.join(save_path, 'model_best.pth.tar'))
    


def load_checkpoint(args, model, model_ema=None, optimizer=None, verbose=True):

    checkpoint = torch.load(args.resume)

    start_epoch = 0
    best_acc = 0

    if "epoch" in checkpoint:
        start_epoch = checkpoint['epoch']

    if "best_acc" in checkpoint:
        best_acc = checkpoint['best_acc']

    model.load_state_dict(checkpoint['state_dict'], False)
    if model_ema:
        model_ema.load_state_dict(checkpoint['state_dict_ema'], False)
        global_step = checkpoint['global_step']

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(args.device)

    if verbose:
        print("=> loading checkpoint '{}' (epoch {})"
                .format(args.resume, start_epoch))
    
    if model_ema:
        return model_ema, optimizer, best_acc, start_epoch
    else:
        return model, optimizer, best_acc, start_epoch