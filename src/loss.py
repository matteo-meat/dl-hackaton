# src/losses.py
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch_geometric.utils import degree

class GCODLoss(Module):
    def __init__(self, lambda_smoothness=1.0):
        super().__init__()
        self.lambda_smoothness = lambda_smoothness

    def forward(self, logits, labels, x, edge_index, batch):
        ce_loss = F.cross_entropy(logits, labels)
        smoothness_loss = self.compute_dirichlet_energy(x, edge_index, batch)
        return ce_loss + self.lambda_smoothness * smoothness_loss

    def compute_dirichlet_energy(self, x, edge_index, batch):
        dirichlet = 0.0
        num_graphs = batch.max().item() + 1

        for i in range(num_graphs):
            mask = (batch == i)
            x_i = x[mask]
            node_mask = mask.nonzero(as_tuple=True)[0]
            edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
            edge_index_i = edge_index[:, edge_mask]

            if x_i.size(0) == 0 or edge_index_i.size(1) == 0:
                continue

            row, col = edge_index_i
            deg = degree(row, x_i.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

            Lx = x_i[row] - x_i[col]
            energy = (norm.view(-1, 1) * (Lx ** 2)).sum()
            dirichlet += energy

        return dirichlet / num_graphs
