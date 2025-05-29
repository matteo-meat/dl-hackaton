
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

from src.utils import set_seed
from src.models import SimpleGCN, CulturalClassificationGNN, GNN, SimpleGIN, SimpleGINE
from src.loadData import GraphDataset


def init_features(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data


def train(data_loader, model, optimizer, criterion, device, save_checkpoints, checkpoint_path, current_epoch):
    model.train()
    total_loss = 0
    for data in tqdm(data_loader, desc="Iterating training graphs", unit="batch"):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if save_checkpoints:
        checkpoint_file = f"{checkpoint_path}_epoch_{current_epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_file)
        print(f"Checkpoint saved at {checkpoint_file}")
    
    return total_loss / len(data_loader)


def evaluate(data_loader, model, criterion, device, calculate_accuracy=False):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    labels = []
    total_val_loss = 0
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Iterating eval graphs", unit="batch"):
            data = data.to(device)
            output = model(data)
            val_loss = criterion(output, data.y)
            total_val_loss += val_loss.item()
            pred = output.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())
            labels.extend(data.y.cpu().numpy())
            if calculate_accuracy:
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)
    if calculate_accuracy:
        accuracy = correct / total
        return predictions, labels, total_val_loss/len(data_loader), accuracy
    return predictions, labels, total_val_loss/len(data_loader)

def save_predictions(predictions, test_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    submission_folder = os.path.join(script_dir, "submission")
    test_dir_name = os.path.basename(os.path.dirname(test_path))
    
    print(f"Making predictions folder at {submission_folder}...")
    os.makedirs(submission_folder, exist_ok=True)
    if os.path.exists(submission_folder):
        print("Submission folder done!")
    
    output_csv_path = os.path.join(submission_folder, f"testset_{test_dir_name}.csv")
    
    print(f"Creating output csv in {output_csv_path}...")
    test_graph_ids = list(range(len(predictions)))
    output_df = pd.DataFrame({
        "id": test_graph_ids,
        "pred": predictions
    })
    
    output_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")

def plot_training_progress(train_losses, train_accuracies, train_f1s, val_losses, val_accuracies, val_f1s, output_dir):
    epochs = range(1, len(train_losses) + 1)
    print("Plotting training process...")
    plt.figure(figsize=(18, 6))

    # Plot training loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label="Training Loss", color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')

    # Plot training accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_accuracies, label="Training Accuracy", color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy per Epoch')

    # Plot training f1
    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_f1s, label="Training F1 score", color='green')
    plt.xlabel('Epoch')
    plt.ylabel('F1 score')
    plt.title('Training F1 score per Epoch')

    # Save plots in the current directory
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_progress.png"))
    plt.close()
    print("Training process plotted!")

    print("Plotting evaluation process...")
    plt.figure(figsize=(18, 6))

    print("Plotting validation loss...")
    # Plot validation loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, val_losses, label="Validation Loss", color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss per Epoch')
    print("Validation loss plotted!")

    print("Plotting validation accuracy...")
    # Plot validation accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, val_accuracies, label="Validation Accuracy", color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy per Epoch')
    print("Validation accuracy plotted!")

    print("Plotting validation f1...")
    # Plot validation f1
    plt.subplot(1, 3, 3)
    plt.plot(epochs, val_f1s, label="Validation F1 score", color='green')
    plt.xlabel('Epoch')
    plt.ylabel('F1 score')
    plt.title('Validation F1 score per Epoch')
    print("Validation f1 plotted!")

    # Save plots in the current directory
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "validation_progress.png"))
    plt.close()
    print("Evaluation process plotted!")


def main(args):

    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Parameters for the GCN model
    input_dim = 300  # Example input feature dimension (you can adjust this)
    hidden_dim = 64
    output_dim = 6  # Number of classes

    script_dir = os.path.dirname(os.path.abspath(__file__))
    num_checkpoints = args.num_checkpoints if args.num_checkpoints else 5

    # Initialize the model, optimizer, and loss criterion
    if args.gnn == 'simple':
         model = SimpleGCN(input_dim, hidden_dim, output_dim).to(device)
    elif args.gnn == 'mnlp':
         model = CulturalClassificationGNN(input_dim, hidden_dim, output_dim).to(device)
    elif args.gnn == 'gin':
        model = GNN(gnn_type = 'gin', num_class = output_dim, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'simple_gin':
        model = SimpleGIN(hidden_dim, output_dim, args.drop_ratio).to(device)
    elif args.gnn == 'simple_gine':
        model = SimpleGINE(hidden_dim, output_dim, args.drop_ratio).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    test_dir_name = os.path.basename(os.path.dirname(args.test_path))

    # Setup logging
    logs_folder = os.path.join(script_dir, "logs", test_dir_name)
    log_file = os.path.join(logs_folder, "training.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())  # Console output as well

    best_checkpoint_path = os.path.join(script_dir, "checkpoints", f"model_{test_dir_name}_best.pth")
    checkpoints_folder = os.path.join(script_dir, "checkpoints", test_dir_name)
    os.makedirs(checkpoints_folder, exist_ok=True)

    # Load pre-trained model for inference
    if os.path.exists(best_checkpoint_path) and not args.train_path:
        model.load_state_dict(torch.load(best_checkpoint_path))
        print(f"Loaded best model from {best_checkpoint_path}")

    # Prepare test dataset and loader
    test_dataset = GraphDataset(args.test_path, transform=init_features)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Train dataset and loader (if train_path is provided)
    if args.train_path:

        train_dataset = GraphDataset(args.train_path, transform=init_features)

        val_ratio = 0.2
        num_val = int(len(train_dataset) * val_ratio)
        num_train = len(train_dataset) - num_val
        train_set, val_set = random_split(train_dataset, [num_train, num_val])

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

        # Training loop
        num_epochs = args.epochs

        train_losses = []
        train_accuracies = []
        train_f1s = []
        val_losses = []
        val_accuracies = []
        val_f1s = []

        best_val_loss = np.inf

        # Calculate intervals for saving checkpoints
        if num_checkpoints > 1:
            checkpoint_intervals = [int((i + 1) * num_epochs / num_checkpoints) for i in range(num_checkpoints)]
        else:
            checkpoint_intervals = [num_epochs]

        patience = args.patience
        epochs_without_improvement = 0
        
        for epoch in range(num_epochs):
            train_loss = train(
                train_loader, model, optimizer, criterion, device,
                save_checkpoints=(epoch + 1 in checkpoint_intervals),
                checkpoint_path=os.path.join(checkpoints_folder, f"model_{test_dir_name}"),
                current_epoch=epoch
            )

            train_preds, train_labels, _, train_acc = evaluate(train_loader, model, criterion, device, calculate_accuracy=True)
            val_preds, val_labels, val_loss, val_acc = evaluate(val_loader, model, criterion, device, calculate_accuracy=True)

            train_f1 = f1_score(train_labels, train_preds, average = "macro")
            val_f1 = f1_score(val_labels, val_preds, average = "macro")

            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

            # Save logs for training progress
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            train_f1s.append(train_f1)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            val_f1s.append(val_f1s)
            logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_checkpoint_path)
                print(f"Best model updated and saved at {best_checkpoint_path}")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs!")
                break
        
        print("Starting plot...")
        plot_training_progress(train_losses, train_accuracies, train_f1s, val_losses, val_accuracies, val_f1s, os.path.join(logs_folder, "plots"))
        print("All plots done!")
    # Evaluate and save test predictions
    print("Loading best model...")
    model.load_state_dict(torch.load(best_checkpoint_path))
    print("Best model loaded!")
    print("Computing predictions...")
    predictions, _, _ = evaluate(test_loader, model, criterion, device, calculate_accuracy=False)
    print("Predictions computed!")
    print("Saving predictions...")
    save_predictions(predictions, args.test_path)
    print("Predictions saved!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate GNN models on graph datasets.")
    parser.add_argument("--train_path", type=str, help="Path to the training dataset (optional).")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
    parser.add_argument("--num_checkpoints", type=int, help="Number of checkpoints to save during training.")
    # parser.add_argument('--device', type=int, default=1, help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='simple', help='GNN simple, mnlp, gin, simple_gin, simple_gine(default: simple)')
    parser.add_argument('--drop_ratio', type=float, default=0.5, help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5, help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300, help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 50)')
    parser.add_argument('--patience', type=int, default=25, help='max number of epochs without training improvements (default: 25)')
    
    args = parser.parse_args()
    main(args)
