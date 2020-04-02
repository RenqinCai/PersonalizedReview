import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

class Reconstruction_loss(nn.Module):
    def __init__(self, device, ignore_index):
        super(Reconstruction_loss, self).__init__()
        self.m_device = device
        self.m_NLL = nn.NLLLoss(size_average=False, ignore_index=ignore_index).to(self.m_device)

    def forward(self, pred, target, length):
        target = target[:, :torch.max(length).item()].contiguous().view(-1)
        pred = pred.view(-1, pred.size(2))

        NLL_loss = self.m_NLL(pred, target)
        return NLL_loss

class KL_loss(nn.Module):
    def __init__(self, device):
        super(KL_loss, self).__init__()
        print("kl loss")

        self.m_device = device

    def forward(self, mean, logv, step, k, x0, anneal_func):
        KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())

        KL_weight = 0
        if anneal_func == "logistic":
            KL_weight = float(1/(1+np.exp(-k*(step-x0))))
        elif anneal_func == "":
            KL_weight = min(1, step/x0)
        else:
            raise NotImplementedError

        return KL_loss, KL_weight

class RRe_loss(nn.Module):
    def __init__(self, device):
        super(RRe_loss, self).__init__()
        self.m_device = device
        self.m_BCE = nn.BCELoss().to(self.m_device)

    def forward(self, pred, target):
        
        RRe_loss = self.m_BCE(pred, target)
        # exit()
        return RRe_loss

class ARe_loss(nn.Module):
    def __init__(self, device):
        super(ARe_loss, self).__init__()
        self.m_device = device
        self.m_BCE = nn.BCELoss().to(self.m_device)
    
    def forward(self, pred, target):
        # print("pred", pred.size())
        # print("target", target.size())

        ARe_loss = self.m_BCE(pred, target)

        return ARe_loss


# class BLEU()