# src/losses.py
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch_geometric.utils import degree

class GCODLoss(Module):
    def __init__(self, lambda_smoothness=1e-2, warmup_epochs: int = 0):
        super().__init__()
        self.base_lambda = lambda_smoothness
        self.warmup_epochs = warmup_epochs

        # Recording the last epoch seen to compute the effective lambda to be apply
        self.last_epoch = 0

    def forward(self, logits, labels, x, edge_index, batch, epoch: int = None, return_components=False):
        ce_loss = F.cross_entropy(logits, labels)
        smoothness_loss = self.compute_dirichlet_energy(x, edge_index, batch)
        
        if self.warmup_epochs > 0 and epoch is not None:
            #Linear ramp from 0 to base_lambda
            frac = min((epoch + 1) / self.warmup_epochs, 1.0)
            lambda_eff = self.base_lambda * frac
        else:
            lambda_eff = self.base_lambda
        
        total_loss = ce_loss + lambda_eff * smoothness_loss

        if return_components:
            return total_loss, ce_loss.detach(), smoothness_loss.detach()
        return total_loss
    
    def compute_dirichlet_energy(self, x, edge_index, batch):
        row, col = edge_index
        N = x.size(0)
        device = x.device
        dtype = x.dtype

        # 1) degree normalization
        deg = degree(row, N, dtype=dtype).to(device)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]             # [E]

        # 2) squared differences
        diff = x[row] - x[col]                                    # [E, F]
        sq = diff.pow(2).sum(dim=1)                               # [E]

        # 3) weighted energy per edge
        energy_e = norm * sq                                      # [E]

        # 4) sum energies per graph
        num_graphs = int(batch.max().item()) + 1
        energy_sum = torch.zeros(num_graphs, device=device).scatter_add_(
            0, batch[row], energy_e
        )  # sums energy_e for each graph index in batch[row]

        # 5) count edges per graph
        edge_counts = torch.zeros(num_graphs, device=device).scatter_add_(
            0, batch[row], torch.ones_like(energy_e)
        )

        # 6) avoid division by zero
        edge_counts = edge_counts.clamp(min=1.0)

        # 7) per‐graph mean, then global mean
        energy_per_graph = energy_sum / edge_counts
        return energy_per_graph.mean()



