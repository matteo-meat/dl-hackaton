# src/losses.py
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch_geometric.utils import degree

class GCODLoss(Module):
    def __init__(self, lambda_smoothness=1.0):
        super().__init__()
        self.lambda_smoothness = lambda_smoothness

    def forward(self, logits, labels, x, edge_index, batch, return_components=False):
        ce_loss = F.cross_entropy(logits, labels)
        smoothness_loss = self.compute_dirichlet_energy(x, edge_index, batch)
        total_loss = ce_loss + self.lambda_smoothness * smoothness_loss

        if return_components:
            return total_loss, ce_loss.detach(), smoothness_loss.detach()
        return total_loss

    def compute_dirichlet_energy(self, x, edge_index, batch):
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype).to(x.device)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        diff = x[row] - x[col]
        if diff.dim() == 1:
            diff = diff.unsqueeze(1)  # ensure shape [E, 1]

        squared_diff = (diff ** 2).sum(dim=1)  # shape [E]
        energy = (norm * squared_diff).sum()

        num_graphs = batch.max().item() + 1
        return energy / num_graphs


