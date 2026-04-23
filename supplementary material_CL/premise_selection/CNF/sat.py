import os
import torch

device = torch.device('cpu')

def bool2sign(b):
    return '' if b else '-'

class CNF:
    def __init__(self, dimacs=None, clauseList=None, device=device):
        self.device = device
        if dimacs is None:
            self.clauseList = [clause(c) for c in clauseList]
            self.atomSet = set().union(*[r.atomSet for r in self.clauseList])
            self.N = len(self.atomSet)
            self.atomList = list(self.atomSet)
            self.atom2idx = {}
            for i in range(self.N):
                self.atom2idx[self.atomList[i]] = i

            self.clauseList = [c for c in self.clauseList if c.atom2sign]

            self.C = self.getC()
            print('\natom2idx:')
            print(self.atom2idx)
        else:
            self.N, self.C = self.parseDimacs(dimacs)
        print('\nThe CNF is represented by a matrix C of shape {}:'.format(self.C.shape))
        print(self.C)

    def getC(self):
        C = torch.zeros([len(self.clauseList), self.N], dtype=torch.float32, device=self.device)
        for rowIdx, c in enumerate(self.clauseList):
            for atom in c.atom2sign:
                colIdx = self.atom2idx[atom]
                C[rowIdx, colIdx] = 1 if c.atom2sign[atom] else -1
        return C

    def parseDimacs(self, dimacs):

        if os.path.isfile(dimacs):
            with open(dimacs, 'r') as dimacs:
                lines = dimacs.readlines()
                lines = [line.strip() for line in lines]

        elif type(dimacs) is str:
            lines = [line.strip() for line in dimacs.split('\n')]
        else:
            assert False, 'Error: the dimacs program is invalid!'
        lines = [line for line in lines if line and not line.startswith('c')]

        line_problem = [line for line in lines if line.startswith('p')]
        assert len(line_problem) == 1, 'Error: there must be exactly 1 line starts with p'
        numVar, numClause = [int(num) for num in line_problem[0].split('cnf')[-1].strip().split()]

        C = torch.zeros([numClause, numVar], dtype=torch.float32, device=self.device)
        for rowIdx, line in enumerate([line for line in lines if not line.startswith('p')]):
            assert line.endswith('0'), 'Error: the line {} does not end with 0'.format(line)
            literals = line[:-1].strip().split()
            signs = [-1 if literal.startswith('-') else 1 for literal in literals]
            atoms = [abs(int(literal))-1 for literal in literals]
            for idx, atom in enumerate(atoms):
                C[rowIdx, atom] = signs[idx]
        return numVar, C

class clause:
    def __init__(self, c):
        self.literals = [literal(l.strip()) for l in c.split('|')]
        self.atomSet = {l.atom for l in self.literals}
        self.atom2sign = self.simplify()

    def __str__(self):
        return ' | '.join([bool2sign(l.sign) + l.atom for l in self.literals])

    def simplify(self):
        atom2sign = {}
        removedAtoms = set()
        for l in self.literals:
            if l.atom in removedAtoms:
                continue
            if l.atom not in atom2sign:
                atom2sign[l.atom] = l.sign
            elif atom2sign[l.atom] == l.sign:
                continue
            else:
                removedAtoms.add(l.atom)
                atom2sign.pop(l.atom, None)
        return atom2sign

class literal:
    def __init__(self, l):
        self.sign = False if l.startswith('-') else True
        self.atom = l if self.sign else l[1:].strip()
    def __str__(self):
        return bool2sign(self.sign) + self.atom
