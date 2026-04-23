import torch
import math

def bp(x):
    return torch.clamp(torch.sign(x-0.5) + 1, max=1)

def binarize(x):
    return torch.clamp(torch.sign(x) + 1, max=1)

def sSTE(grad_output, x=None):
    return grad_output * (torch.le(x, 1) * torch.ge(x, -1)).float()

def tSTE(grad_output, x=None):
    return grad_output * ((exp(x)-exp(-x))/(exp(x)+exp(-x))).float()

class Disc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return bp(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
B = Disc.apply

class DiscBi(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return binarize(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
Bi = DiscBi.apply

class DiscBs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return binarize(x)
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = sSTE(grad_output, x)
        return grad_input
Bs = DiscBs.apply

class DiscBs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return binarize(x)
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = tSTE(grad_output, x)
        return grad_input
Bs = DiscBs.apply

def one(x):
    return (x == 1).int()

def minusOne(x):
    return (x == -1).int()

def zero(x):
    return (x == 0).int()

def reg_bound(output):
    return output.pow(2).mean()

def reg_cnf(C, v, g):
    if len(C.shape) == 2:
        C = C.unsqueeze(0)

    v, g = v.unsqueeze(1), g.unsqueeze(1)
    L_v, L_g = one(C) * v + minusOne(C) * (1-v), C * g
    unsat = (1 - L_v).prod(dim=-1)
    up = (C.abs().sum(dim=-1) - minusOne(L_g).sum(dim=-1)) == 1

    keep = (one(L_v) * (1-L_v) + zero(L_v) * L_v).sum(dim=-1)
    if (unsat==1).any():
        L_unsat = unsat[unsat==1].mean()
    else:
        L_unsat = torch.tensor(0.0, requires_grad=True)
    if (unsat==0).any():
        L_sat = keep[unsat==0].mean()
    else:
        L_sat = torch.tensor(0.0, requires_grad=True)

    L_up = unsat[up].sum(dim=-1).mean()
    return L_sat + L_unsat + L_up
