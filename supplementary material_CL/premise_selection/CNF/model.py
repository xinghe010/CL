import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_scatter import scatter_mean
from torch_geometric.nn import global_mean_pool
from sat import CNF
from ste import reg_bound, reg_cnf
from add import read_cnf
from custom_modules import EWGS_discretizer

class MLPBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 activation=F.relu,
                 bias=True,
                 batch=False,
                 drop=False
 ):
        super(MLPBlock, self).__init__()
        self.activation = activation
        self.bias = bias
        self.batch = batch
        self.drop = drop
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)
        if batch:
            self.BN = nn.BatchNorm1d(out_channels)
        if self.drop:
            self.drop = nn.Dropout(0.2)
        self.reset_parameters()

    def reset_parameters(self):
        if self.activation == F.relu:
            nn.init.kaiming_normal_(self.lin.weight, nonlinearity="relu")
        elif self.activation == F.leaky_relu:
            nn.init.kaiming_normal_(self.lin.weight)
        else:
            nn.init.xavier_normal_(self.lin.weight)
        if self.bias:
            nn.init.zeros_(self.lin.bias)

    def forward(self, x):
        x = self.lin(x)
        if self.batch and x.size()[0] > 1:
            x = self.BN(x)
        if self.drop:
            x = self.drop(x)
        x = self.activation(x)
        return x

class Initialization(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Initialization, self).__init__()
        self.embedding = nn.Embedding(in_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.embedding.weight)

    def forward(self, x):
        indices = torch.argmax(x, dim=1)
        output = self.embedding(indices)
        return output

class DAGEmbedding(nn.Module):
    def __init__(self, node_out_channels, layers):
        super(DAGEmbedding, self).__init__()
        self.K = layers
        self.F_T = nn.ModuleList([
            MLPBlock(3 * node_out_channels, node_out_channels, batch=True)
            for i in range(layers)
        ])
        self.F_M = nn.ModuleList([
            MLPBlock(3 * node_out_channels, node_out_channels, batch=True)
            for i in range(layers)
        ])
        self.F_B = nn.ModuleList([
            MLPBlock(3 * node_out_channels, node_out_channels, batch=True)
            for i in range(layers)
        ])
        self.F_trans_TW = nn.ModuleList([
            MLPBlock(node_out_channels, node_out_channels, batch=True)
            for i in range(layers)
        ])

    def forward(self, x, term_walk_index):
        N = x.size()[0]
        for i in range(self.K):
            term_walk_feat = torch.cat([x[term_walk_index[0]],
                                        x[term_walk_index[1]],
                                        x[term_walk_index[2]]], dim=1)

            trans_T = self.F_T[i](term_walk_feat)
            m_T = scatter_mean(trans_T,
                               index=term_walk_index[0],
                               dim=0, dim_size=N)

            trans_M = self.F_M[i](term_walk_feat)
            m_M = scatter_mean(trans_M,
                               index=term_walk_index[1],
                               dim=0, dim_size=N)

            trans_B = self.F_B[i](term_walk_feat)
            m_B = scatter_mean(trans_B,
                               index=term_walk_index[2],
                               dim=0, dim_size=N)

            m_TW = m_T + m_M + m_B
            m_TW = self.F_trans_TW[i](m_TW)
            x = x + m_TW
        return x

class DAGPooling(nn.Module):
    def __init__(self):
        super(DAGPooling, self).__init__()

    def forward(self, x, batch_index):
        output = global_mean_pool(x, batch_index)
        return output

class Classifier(nn.Module):
    def __init__(self, node_out_channels):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            MLPBlock(2 * node_out_channels, node_out_channels,
                     batch=True),
            MLPBlock(node_out_channels, 2, activation=lambda x: x,
                     batch=True))

    def forward(self, conj_batch, prem_batch):
        x_concat = torch.cat([conj_batch, prem_batch], dim=1)
        pred_y = self.classifier(x_concat)
        return pred_y

class PremiseSelectionModel(nn.Module):
    def __init__(self, node_in_channels, node_out_channels, layers, reg, device,
                 weighting="sum"):
        super(PremiseSelectionModel, self).__init__()
        self.initial = Initialization(node_in_channels, node_out_channels)
        self.dag_emb = DAGEmbedding(node_out_channels, layers)
        self.pooling = DAGPooling()
        self.classifier = Classifier(node_out_channels)
        self.criterion = nn.CrossEntropyLoss()
        self.corrects = None
        self.cross, self.cnf, self.bound = [s in reg for s in ('cross', 'cnf', 'bound')]

        self.weighting = weighting
        if weighting == "uncertainty":
            self.log_vars = nn.Parameter(torch.zeros(3))
        self.C, self.atom2idx = read_cnf('./add.cnf', './add.atom2idx')
        self.C = self.C.to(device)
        self.numClauses, self.numAtoms = self.C.shape
        self.buff_out = None
        self.bkwd_scaling_factorO = 0.0
        self.EWGS_discretizer = EWGS_discretizer.apply
        self.hook_Qvalues = False

        self.register_buffer('bkwd_scaling_factor', torch.tensor(self.bkwd_scaling_factorO).float())

    def out_quantization(self, x):

        x = self.EWGS_discretizer(x, self.bkwd_scaling_factor)

        if self.hook_Qvalues:
            self.buff_out = x
            self.buff_out.retain_grad()

        return x

    def initialize(self, x):

        Qout = x

        self.uO.data.fill_(x.std() / math.sqrt(1 - 2 / math.pi) * 3.0)
        self.lO.data.fill_(x.min())
        Qout = self.out_quantization(x)

    def forward(self, batch, device, hyper):
        h_s = self.initial(batch.x_s)
        h_t = self.initial(batch.x_t)
        h_s = self.dag_emb(h_s, batch.term_walk_index_s)
        h_t = self.dag_emb(h_t, batch.term_walk_index_t)
        h_g_s = self.pooling(h_s, batch.x_s_batch)
        h_g_t = self.pooling(h_t, batch.x_t_batch)
        pred_y = self.classifier(h_g_s, h_g_t)
        pred_label = torch.max(pred_y, dim=1)[1]
        self.corrects = (pred_label == batch.y).sum().cpu().item()

        Qout = pred_y
        Qout = self.out_quantization(Qout)

        losses = []

        if self.cross:
            losses.append(self.criterion(Qout, batch.y))
        if self.cnf:
            v, g = self.vg_gen(Qout, batch.y, device)

            losses.append(reg_cnf(self.C, v, g) * hyper[0])
        if self.bound:
            pred_y_b = Qout[:]

            losses.append(reg_bound(pred_y_b) * hyper[1])

        if self.weighting == "mean":
            loss = sum(losses) / max(len(losses), 1)
        elif self.weighting == "uncertainty":
            stacked = []
            for i, l in enumerate(losses):
                lv = self.log_vars[i]
                stacked.append(torch.exp(-lv) * l + lv)
            loss = sum(stacked)
        else:
            loss = sum(losses)
        return loss, batch.y, pred_label

    def vg_gen(self, Qout, y, device):
        batch_size = y.shape[0]
        g = torch.zeros((batch_size, self.numAtoms), device=device)
        v = torch.zeros((batch_size, self.numAtoms), device=device)

        g[:,-2:] = F.one_hot(y, num_classes=2)

        Qout = torch.sigmoid(Qout)

        v[:,:2]= Qout

        v = g + v * (g==0).int()

        return v, g
