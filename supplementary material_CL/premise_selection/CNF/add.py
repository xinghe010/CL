import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, Subset
from torchvision.transforms import transforms

from sat import CNF
from ste import B, reg_bound, reg_cnf

def write_cnf(path_cnf, path_atom2idx):

    atom2idx = {}
    idx = 1

    for l in range(2):
        atom2idx[f'predict({l})'] = idx
        idx += 1

    for s in range(11):
        atom2idx[f'bl({s})'] = idx
        idx += 1

    numClauses = 0
    cnf2 = ''

    for l in range(11):
        atom = atom2idx[f'bl({s})']
        atoms = [str(atom2idx[f'predict({l})']) for l in range(2)]
        atoms = ' '.join(atoms)
        cnf2 += f'{atoms} -{atom} 0\n'
        numClauses += 1

    cnf1 = f'p cnf {idx-1} {numClauses}\n'
    with open(path_cnf, 'w') as f:
        f.write(cnf1 + cnf2)
    json.dump(atom2idx, open(path_atom2idx,'w'))
    return atom2idx

def read_cnf(path_cnf, path_atom2idx):
    try:
        cnf = CNF(dimacs=path_cnf)
        atom2idx = json.load(open(path_atom2idx))
    except:
        atom2idx = write_cnf(path_cnf, path_atom2idx)
        cnf = CNF(dimacs=path_cnf)
    return cnf.C, atom2idx
