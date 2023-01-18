# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import torch
import torch.nn as nn


class MeanTeacherModel(nn.Module):
    """
    Build a Mean Teacher Model with: a teacher encoder, a student encoder
    """
    def __init__(self, base_encoder, args):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MeanTeacherModel, self).__init__()

        self.consistency = args.consistency
        self.consistency_rampup = args.consistency_rampup
        self.ema_decay = args.ema_decay
        

        # create the encoders
        # num_classes is the output fc dimension
        self.teacher = base_encoder(args)
        self.student = base_encoder(args)

        for param_s, param_t in zip(self.student.parameters(), self.teacher.parameters()):
            param_t.data.copy_(param_s.data)  # initialize
            param_t.detach_()
            param_t.requires_grad = False  # not update by gradient

        # create the queue
        # self.register_buffer("global_step", torch.tensor(0, dtype=torch.float32))

    @torch.no_grad()
    def _ema_update_teacher(self):
        """
        Momentum update of the key encoder
        """
        # for param_q, param_k in zip(self.student.parameters(), self.teacher.parameters()):
            # param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        
        self.global_step += 1
        alpha = min(1 - 1 / (self.global_step + 1), self.ema_decay)
        for param_t, param_s in zip(self.teacher.parameters(), self.student.parameters()):
            param_t.data.mul_(alpha).add_(1 - alpha, param_s.data)
            
            
    @torch.no_grad()
    def _sigmoid_rampup(self, current):
        if self.consistency_rampup == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, self.consistency_rampup)
            phase = 1.0 - current / self.consistency_rampup
            return float(np.exp(-5.0 * phase * phase))
    
    @torch.no_grad()
    def get_current_consistency_weight(self, epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return self.consistency * self._sigmoid_rampup(epoch)


    def forward(self, im_s, im_t):

        output_s = self.student(im_s)
        
        with torch.no_grad():
            # self._ema_update_teacher()
            output_t = self.teacher(im_t)

        return output_s, output_t