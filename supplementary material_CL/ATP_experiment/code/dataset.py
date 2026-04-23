import torch
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
from torch.nn.functional import one_hot
import json

from formula_parser import fof_formula_transformer
from graph import Graph
from utils import read_file, load_pickle_file, Statements

def flatten(sequence):
    for item in sequence:
        if type(item) is list:
            for subitem in flatten(item):
                yield subitem
        else:
            yield item

class PairData(Data):
    def __init__(self, x_s=None, term_walk_index_s=None,
                 x_t=None, term_walk_index_t=None, y=None, similar=None):
        super().__init__()
        self.x_s = x_s
        self.x_t = x_t
        self.term_walk_index_s = term_walk_index_s
        self.term_walk_index_t = term_walk_index_t
        self.y = y

    def __inc__(self, key, value, *args, **kwargs):
        if key == "term_walk_index_s":
            return self.x_s.size(0)
        if key == "term_walk_index_t":
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

class FormulaGraphDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 data_class,
                 statements_file,
                 node_dict_file,
                 rename=True):
        self.root = root
        self.data_class = data_class
        self.statements = Statements(statements_file)
        self.rename = rename
        self.node_dict = load_pickle_file(node_dict_file)
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["{}.json".format(self.data_class)]

    @property
    def processed_file_names(self):
        return ["{}.pt".format(self.data_class)]

    def graph_process(self, G):
        nodes = []
        term_walk_indices = []

        for node in G:
            nodes.append(node.name)
            if node.parents and node.children:
                for parent in node.parents:
                    for child in node.children:
                        term_walk_indices.append([parent.id,
                                                  node.id,
                                                  child.id])

        term_walk_indices = np.array(
            term_walk_indices, dtype=np.int64).reshape(-1, 3).T

        return nodes, term_walk_indices

    def vectorization(self, objects, object_dict):
        indices = [object_dict[obj] for obj in objects]
        onehot = one_hot(torch.LongTensor(indices), len(object_dict)).float()
        return onehot

    def similar(self, conj, prem):
        a1 = []
        b1 = []
        n = []
        for x in flatten(conj):
            a1.append(x)
        for x in flatten(prem):
            b1.append(x)
        for i in b1:
            if i in a1:
                n.append(i)
        s = len(n) / (len(b1))
        if s > 0.6:
            return 1
        else:
            return 0

    def process(self):
        raw_examples = \
            [json.loads(line) for line in read_file(self.raw_paths[0])]
        dataList = []

        for example in raw_examples:

            conj, prem, label = example
            conj_graph = Graph(fof_formula_transformer(self.statements[conj]),
                               rename=self.rename)

            prem_graph = Graph(fof_formula_transformer(self.statements[prem]),
                               rename=self.rename)
            conj_formula = fof_formula_transformer(self.statements[conj])
            prem_formula = fof_formula_transformer(self.statements[conj])
            s = self.similar(conj_formula, prem_formula)

            c_nodes, c_term_walk_indices = self.graph_process(conj_graph)
            p_nodes, p_term_walk_indices = self.graph_process(prem_graph)
            data = PairData(
                x_s=self.vectorization(c_nodes, self.node_dict),
                term_walk_index_s=torch.from_numpy(c_term_walk_indices),
                x_t=self.vectorization(p_nodes, self.node_dict),
                term_walk_index_t=torch.from_numpy(p_term_walk_indices),
                y=torch.LongTensor([label]),
                similar=torch.LongTensor(s))
            dataList.append(data)
        data, slices = self.collate(data_list=dataList)
        torch.save((data, slices), self.processed_paths[0])
