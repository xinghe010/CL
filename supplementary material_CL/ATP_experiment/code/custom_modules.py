import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class EWGS_discretizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scaling_factor):

        x_out = torch.round(x)
        ctx._scaling_factor = scaling_factor
        ctx.save_for_backward(x-x_out)
        return x_out

    @staticmethod
    def backward(ctx, g):
        diff = ctx.saved_tensors[0]
        delta = ctx._scaling_factor
        scale = 1 + delta * torch.sign(g)*diff
        return g * scale, None, None
