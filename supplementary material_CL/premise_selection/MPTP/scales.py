import numpy as np
import torch
import sys
from tqdm import tqdm
from model import PremiseSelectionModel
from model import Classifier
__all__ = [ 'update_grad_scales']

def update_grad_scales(model, train_loader, device, args, hyper):

    if args.QoutFlag:
        scaleO = []
    for m in model.modules():
        if isinstance(m, PremiseSelectionModel):
            m.hook_Qvalues = True
            if args.QoutFlag:
                scaleO.append(0)

    model.train()
    with tqdm(total=4, file=sys.stdout) as pbar:
        for num_batches, batch in enumerate(train_loader):
            if num_batches == 3:
                break

            model.zero_grad()
            batch = batch.to(device)
            loss, batch.y, pred_label = model(batch, device, hyper)
            loss.backward(create_graph=True)

            if args.QoutFlag:
                Qout = []
            for m in model.modules():
                if isinstance(m, PremiseSelectionModel):
                    if args.QoutFlag:
                        Qout.append(m.buff_out)

            if args.QoutFlag:
                params = []
                grads = []
                for i in range(len(Qout)):
                    params.append(Qout[i])
                    grads.append(Qout[i].grad)
                    model.zero_grad()

                for i in range(len(Qout)):
                    trace_hess_O = np.mean(trace(model, [params[i]], [grads[i]], device))
                    avg_trace_hess_O = trace_hess_O / params[i].view(-1).size()[0]
                    scaleO[i] += (avg_trace_hess_O / (grads[i].std().cpu().item() * 3.0))
            pbar.update(1)

    if args.QoutFlag:
        for i in range(len(scaleO)):
            scaleO[i] /= num_batches
            scaleO[i] = np.clip(scaleO[i], 0, np.inf)
        print("\n\nscaleO\n", scaleO)
    print("")
    i = 0
    for m in model.modules():
        if isinstance(m, PremiseSelectionModel):
            if args.QoutFlag:
                m.bkwd_scaling_factor.data.fill_(scaleO[i])
            m.hook_Qvalues = False
            i += 1

def group_product(xs, ys):
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])

def hessian_vector_product(gradsH, params, v):
    hv = torch.autograd.grad(gradsH,
                             params,
                             grad_outputs=v,
                             only_inputs=True,
                             retain_graph=True)
    return hv

def trace(model, params, grads, device, maxIter=50, tol=1e-3):

    trace_vhv = []
    trace = 0.

    for i in range(maxIter):
        model.zero_grad()
        v = [
            torch.randint_like(p, high=2, device=device)
            for p in params
        ]

        for v_i in v:
            v_i[v_i == 0] = -1

        Hv = hessian_vector_product(grads, params, v)
        trace_vhv.append(group_product(Hv, v).cpu().item())
        if abs(np.mean(trace_vhv) - trace) / (trace + 1e-6) < tol:
            return trace_vhv
        else:
            trace = np.mean(trace_vhv)

    return trace_vhv
