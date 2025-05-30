
import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

import os
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from src.utils import set_seed, plot_training_progress, save_predictions
from src.models import SimpleGCN, GNN, SimpleGINE, EnhancedGINEWithVN
from src.loadData import GraphDataset
from src.loss import GCODLoss


def init_features(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data

def train(
    data_loader, model, optimizer, criterion, device,
    save_checkpoints, checkpoint_path, current_epoch,
    use_gcod=False
    ):

    model.train()
    total_loss = 0
    total_ce = 0
    total_dirichlet = 0

    for data in tqdm(data_loader, desc="Iterating training graphs", unit="batch"):
        data = data.to(device)
        optimizer.zero_grad()

        if use_gcod:
            output, node_embed = model(data)
            loss, ce, dirichlet = criterion(output, data.y, node_embed, data.edge_index, data.batch, epoch=current_epoch ,return_components=True)
            total_ce  += ce.item()
            total_dirichlet += dirichlet.item()
        else:
            output, _ = model(data)
            loss = criterion(output, data.y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Save checkpoint if needed
    if save_checkpoints:
        ckpt = f"{checkpoint_path}_epoch_{current_epoch + 1}.pth"
        torch.save(model.state_dict(), ckpt)
        print(f"Checkpoint saved at {ckpt}")

    n = len(data_loader)
    avg_loss = total_loss / n
    avg_ce = total_ce / n if use_gcod else 0.0
    avg_dirichlet = total_dirichlet / n if use_gcod else 0.0

    return avg_loss, avg_ce, avg_dirichlet


def evaluate_training(data_loader, model, criterion, device, use_gcod=False):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    labels = []
    total_val_loss = 0
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Iterating eval graphs", unit="batch"):
            data = data.to(device)
            
            if use_gcod:
                output, node_embeddings = model(data)
                val_loss, _, _ = criterion(output, data.y, node_embeddings, data.edge_index, data.batch, return_components=True)
            else:
                output, _ = model(data)
                val_loss = criterion(output, data.y)
                
            total_val_loss += val_loss.item()
            pred = output.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())
            labels.extend(data.y.cpu().numpy())
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)
    
    accuracy = correct / total
    return predictions, labels, total_val_loss / len(data_loader), accuracy

def evaluate_testing(data_loader, model, device, use_gcod=False):
    model.eval()
    predictions = []
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Iterating test graphs", unit="batch"):
            data = data.to(device)
            if use_gcod:
                output, _ = model(data)
            else:
                output, _ = model(data)
            pred = output.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())
    return predictions


def main(args):
    # ========================
    # INITIALIZATION
    # ========================
    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model parameters
    input_dim = 300  # Example input feature dimension (you can adjust this)
    hidden_dim = 64
    output_dim = 6  # Number of classes
    
    # Directory setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir_name = os.path.basename(os.path.dirname(args.test_path))
    num_checkpoints = args.num_checkpoints if args.num_checkpoints else 5
    
    # ========================
    # MODEL INITIALIZATION
    # ========================
    if args.gnn == 'simple':
         model = SimpleGCN(input_dim, hidden_dim, output_dim).to(device)
    elif args.gnn == 'ena_gine':
         model = EnhancedGINEWithVN(hidden_dim, output_dim).to(device)
    elif args.gnn == 'gin':
        model = GNN(gnn_type = 'gin', num_class = output_dim, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'simple_gine':
        model = SimpleGINE(hidden_dim, output_dim, args.drop_ratio).to(device)
    
    if args.use_gcod:
        criterion = GCODLoss(lambda_smoothness=args.gcod_lambda, warmup_epochs=args.gcod_warmup_epochs)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    
    # ========================
    # LOGGING AND CHECKPOINT SETUP
    # ========================
    logs_folder = os.path.join(script_dir, "logs", test_dir_name)
    log_file = os.path.join(logs_folder, "training.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())  # Console output as well

    best_checkpoint_path = os.path.join(script_dir, "checkpoints", f"model_{test_dir_name}_best.pth")
    checkpoints_folder = os.path.join(script_dir, "checkpoints", test_dir_name)
    os.makedirs(checkpoints_folder, exist_ok=True)

    # ========================
    # LOAD PRE-TRAINED MODEL (IF EXISTS)
    # ========================
    if os.path.exists(best_checkpoint_path) and not args.train_path:
        model.load_state_dict(torch.load(best_checkpoint_path))
        print(f"Loaded best model from {best_checkpoint_path}")

    # ========================
    # DATA PREPARATION
    # ========================
    # Test dataset and loader
    test_dataset = GraphDataset(args.test_path, transform=init_features)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # ========================
    # TRAINING (IF TRAIN_PATH PROVIDED)
    # ========================
    if args.train_path:
        # Training dataset preparation
        train_dataset = GraphDataset(args.train_path, transform=init_features)

        val_ratio = 0.2
        num_val = int(len(train_dataset) * val_ratio)
        num_train = len(train_dataset) - num_val
        train_set, val_set = random_split(train_dataset, [num_train, num_val])

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

        # Training parameters
        num_epochs = args.epochs
        patience = args.patience
        epochs_without_improvement = 0
        best_val_loss = np.inf
        best_val_f1 = 0
        
        # Training metrics tracking
        train_losses = []
        train_accuracies = []
        train_f1s = []
        val_losses = []
        val_accuracies = []
        val_f1s = []

        # Calculate checkpoint intervals
        if num_checkpoints > 1:
            checkpoint_intervals = [int((i + 1) * num_epochs / num_checkpoints) for i in range(num_checkpoints)]
        else:
            checkpoint_intervals = [num_epochs]
        
        # Training loop
        for epoch in range(num_epochs):
            # Training step
            train_loss, train_ce, train_dirichlet = train(
                train_loader, model, optimizer, criterion, device,
                save_checkpoints=(epoch + 1 in checkpoint_intervals),
                checkpoint_path=os.path.join(checkpoints_folder, f"model_{test_dir_name}"),
                current_epoch=epoch,
                use_gcod=args.use_gcod
            )

            # Evaluation step
            train_preds, train_labels, _, train_acc = evaluate_training(train_loader, model, criterion, device, use_gcod=args.use_gcod)
            val_preds, val_labels, val_loss, val_acc = evaluate_training(val_loader, model, criterion, device, use_gcod=args.use_gcod)
            train_f1 = f1_score(train_labels, train_preds, average = "macro")
            val_f1 = f1_score(val_labels, val_preds, average = "macro")

            # Save metrics for plotting
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            train_f1s.append(train_f1)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            val_f1s.append(val_f1)
            logging.info(
                f"Epoch {epoch + 1}/{num_epochs}\n"
                f"  Train -> Loss: {train_loss:.4f} | CE: {train_ce:.4f} | Dirichlet: {train_dirichlet:.4f} | "
                f"Acc: {train_acc:.4f} | F1: {train_f1:.4f}\n"
                f"  Val   -> Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}"
            )
                        
            # Best model saving and early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_checkpoint_path)
                print(f"✅ Best model updated (Val Acc: {val_acc:.4f}) and saved at {best_checkpoint_path}✅")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                print(f"No improvement in validation accuracy for {epochs_without_improvement} epoch(s)")

            if epochs_without_improvement >= patience:
                print(f"🛑 Early stopping triggered after {epoch + 1} epochs!🛑")
                break
        
        # Plot training progress
        plot_training_progress(train_losses, train_accuracies, train_f1s, val_losses, val_accuracies, val_f1s, os.path.join(logs_folder, "plots"))

    # ========================
    # FINAL EVALUATION AND PREDICTION
    # ========================
    model.load_state_dict(torch.load(best_checkpoint_path))
    predictions = evaluate_testing(test_loader, model, device)
    save_predictions(predictions, args.test_path)


if __name__ == "__main__":
    num_epochs = 700
    patience = 25
    batch_size = 128
    use_gcod = False
    gcod_lambda = 1e-4
    warmup_epochs = 30

    parser = argparse.ArgumentParser(description="Train and evaluate GNN models on graph datasets.")
    parser.add_argument("--train_path", type=str, help="Path to the training dataset (optional).")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
    parser.add_argument("--num_checkpoints", type=int, help="Number of checkpoints to save during training.")
    # parser.add_argument('--device', type=int, default=1, help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='simple', help='GNN simple, mnlp, gin, simple_gin, simple_gine(default: simple)')
    parser.add_argument('--drop_ratio', type=float, default=0.5, help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5, help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300, help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=batch_size, help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=num_epochs, help='number of epochs to train (default: 50)')
    parser.add_argument('--patience', type=int, default=patience, help='max number of epochs without training improvements (default: 25)')
    parser.add_argument('--use_gcod', action='store_true', default=use_gcod, help='Use GCOD loss instead of CrossEntropyLoss')
    parser.add_argument('--gcod_lambda', type=float, default=gcod_lambda,help="final smoothness weight λ (default: 1e-2)")
    parser.add_argument("--gcod_warmup_epochs", type=int, default=warmup_epochs)

    args = parser.parse_args()
    main(args)
