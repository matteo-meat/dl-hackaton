import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):

    def __init__(self, alpha = 1, gamma = 2, reduction = "mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

class GCODLoss(torch.nn.Module):
    def __init__(self, num_samples, num_classes, device, u_lr=1.0):
        super().__init__()
        self.u = torch.full((num_samples,), 0.5, device=device, requires_grad=False)
        self.num_classes = num_classes
        self.u_lr = u_lr
        self.a_train = 0.0  # Updated after each epoch
        self.num_samples = num_samples
        self.eps = 1e-7  # Small epsilon for numerical stability

    def forward(self, outputs, labels, indices):
        
        if indices is None:
            return torch.tensor(0.0), torch.tensor(0.0)
        
        indices = torch.clamp(indices, 0, self.num_samples - 1)

        u_B = self.u[indices].detach()
        y_B = F.one_hot(labels, num_classes=self.num_classes).float()

        # Compute soft labels y_tilde_B
        term1 = (1 - u_B.unsqueeze(1)) * y_B
        term2 = u_B.unsqueeze(1) * (1 - y_B) / max(self.num_classes - 1, 1)
        y_tilde_B = term1 + term2

        # L1: Adjusted outputs and cross-entropy
        adjusted_outputs = outputs + self.a_train * u_B.unsqueeze(1) * y_B
        log_probs = F.log_softmax(adjusted_outputs, dim=1)
        L1 = - (y_tilde_B * log_probs).sum(dim=1)
        L1 = L1.mean()

        # L2: Prediction consistency term
        preds = torch.argmax(outputs, dim=1)
        y_hat_B = F.one_hot(preds, num_classes=self.num_classes).float()
        diff = y_hat_B + u_B.unsqueeze(1) * y_B - y_B
        L2 = (1 / self.num_classes) * (diff ** 2).sum(dim=1).mean()

        # L3: KL divergence regularization
        true_logits = (outputs * y_B).sum(dim=1)
        p = torch.sigmoid(true_logits).clamp(self.eps, 1 - self.eps)
        u_B_clamped = u_B.clamp(self.eps, 1 - self.eps)
        q = torch.sigmoid(-torch.log(u_B_clamped)).clamp(self.eps, 1 - self.eps)

        P = torch.stack([p, 1 - p], dim=1)
        Q = torch.stack([q, 1 - q], dim=1)

        L3 = (1 - self.a_train) * F.kl_div(
            torch.log(P), 
            Q, 
            reduction='batchmean',
            log_target=False
        )

        total_loss = L1 + L3
        return total_loss, L2

    def update_u(self, indices, L2_grad):
        if indices is None:
            return
        
        with torch.no_grad():
            indices = torch.clamp(indices.long(), 0, self.num_samples - 1)
            self.u[indices] -= self.u_lr * L2_grad
            self.u[indices] = self.u[indices].clamp(0, 1)

    def set_a_train(self, a_train):
        self.a_train = a_train