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
        self.u = torch.zeros(num_samples, device=device, requires_grad=False)
        self.num_classes = num_classes
        self.u_lr = u_lr
        self.a_train = 0.0  # Updated after each epoch
        self.num_samples = num_samples

    def forward(self, outputs, labels, indices):
        indices = indices.long()
        indices = torch.clamp(indices, 0, self.num_samples - 1)

        u_B = self.u[indices].detach()
        y_B = F.one_hot(labels, num_classes=self.num_classes).float()

        # Compute soft labels y_tilde_B
        term1 = (1 - u_B.unsqueeze(1)) * y_B
        term2 = u_B.unsqueeze(1) * (1 - y_B) / (self.num_classes - 1)
        y_tilde_B = term1 + term2

        # L1: Adjusted outputs and cross-entropy
        adjusted_outputs = outputs + self.a_train * u_B.unsqueeze(1) * y_B
        L1 = - (y_tilde_B * F.log_softmax(adjusted_outputs, dim=1)).sum(dim=1).mean()

        # L2: Prediction consistency term
        preds = torch.argmax(outputs, dim=1)
        y_hat_B = F.one_hot(preds, num_classes=self.num_classes).float()
        diff = y_hat_B + u_B.unsqueeze(1) * y_B - y_B
        L2 = (1 / self.num_classes) * (diff ** 2).sum(dim=1).mean()

        # L3: KL divergence regularization
        true_logits = (outputs * y_B).sum(dim=1)
        p = torch.sigmoid(true_logits)
        u_B_clamped = u_B.clamp(min=1e-8, max=1-1e-8)
        q = torch.sigmoid(-torch.log(u_B_clamped))
        
        term1 = p * torch.log(p / q)
        term2 = (1 - p) * torch.log((1 - p) / (1 - q))
        L3 = (1 - self.a_train) * (term1 + term2).mean()

        total_loss = L1 + L3
        return total_loss, L2

    def update_u(self, indices, L2_grad):
        with torch.no_grad():
            indices = torch.clamp(indices.long(), 0, self.num_samples - 1)
            self.u[indices] -= self.u_lr * L2_grad
            self.u[indices] = self.u[indices].clamp(0, 1)

    def set_a_train(self, a_train):
        self.a_train = a_train