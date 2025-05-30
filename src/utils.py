import torch
import random
import numpy as np
import tarfile
import os
import matplotlib.pyplot as plt
import pandas as pd

def set_seed(seed=777):
    seed = seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed()

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

    plt.figure(figsize=(18, 6))

    # Plot validation loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, val_losses, label="Validation Loss", color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss per Epoch')
    print("Validation loss plotted!")

    # Plot validation accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, val_accuracies, label="Validation Accuracy", color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy per Epoch')

    # Plot validation f1
    plt.subplot(1, 3, 3)
    plt.plot(epochs, val_f1s, label="Validation F1 score", color='green')
    plt.xlabel('Epoch')
    plt.ylabel('F1 score')
    plt.title('Validation F1 score per Epoch')

    # Save plots in the current directory
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "validation_progress.png"))
    plt.close()