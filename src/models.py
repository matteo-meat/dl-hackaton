import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, GINEConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GraphNorm

class DefaultGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DefaultGCN, self).__init__()
        self.embedding = torch.nn.Embedding(1, input_dim) 
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.global_pool = global_mean_pool  
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.embedding(x)  
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = self.global_pool(x, batch)  
        out = self.fc(x)  
        return out

class DefaultGIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DefaultGIN, self).__init__()
        self.embedding = torch.nn.Embedding(1, input_dim) 
        self.conv1 = GINConv(nn.Sequential(torch.nn.Linear(input_dim, hidden_dim), torch.nn.ReLU(), torch.nn.Linear(hidden_dim, hidden_dim)))
        self.conv2 = GINConv(nn.Sequential(torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(), torch.nn.Linear(hidden_dim, hidden_dim)))
        self.global_pool = global_mean_pool  
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.embedding(x)  
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = self.global_pool(x, batch)  
        out = self.fc(x)  
        return out
    
class SimpleGIN(torch.nn.Module):

    def __init__(self, hidden_dim, output_dim):
        super(SimpleGIN, self).__init__()

        self.node_embedding = nn.Embedding(1, hidden_dim)

        nn1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), 
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim))
        self.conv1 = GINConv(nn1)

        nn2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), 
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim))
        
        self.conv2 = GINConv(nn2)

        self.lin = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.node_embedding(x)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        x = global_mean_pool(x, batch)

        x = self.lin(x)

        return x
    
class SimpleGINE(torch.nn.Module):

    def __init__(self, hidden_dim, output_dim):
        super(SimpleGINE, self).__init__()

        self.node_embedding = nn.Embedding(1, hidden_dim)

        nn1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), 
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim))
        self.conv1 = GINEConv(nn1, edge_dim = 7)

        nn2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), 
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim))
        
        self.conv2 = GINEConv(nn2, edge_dim = 7)

        self.lin = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.node_embedding(x)

        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)

        x = global_mean_pool(x, batch)

        x = self.lin(x)

        return x
    
class TurboGNN(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=5, num_classes=6):
        super().__init__()
        # Input embedding
        self.embedding = nn.Embedding(1, hidden_dim)
        
        # Edge feature projector
        self.edge_encoder = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Convolutional layers with residual connections
        self.convs = nn.ModuleList()
        self.gnorms = nn.ModuleList()
        for i in range(num_layers):
            conv = GINEConv(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim * 2, hidden_dim)
                ), edge_dim=hidden_dim, train_eps=True
            )
            self.convs.append(conv)
            self.gnorms.append(GraphNorm(hidden_dim))
        
        # Context-aware attention pooling
        self.att_pool = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Classifier with skip connections
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Encode features
        x = self.embedding(x)
        edge_emb = self.edge_encoder(edge_attr)
        
        # Message passing with residuals
        features = []
        for i, (conv, gnorm) in enumerate(zip(self.convs, self.gnorms)):
            residual = x
            x = conv(x, edge_index, edge_emb)
            x = gnorm(x)
            x = F.relu(x)
            x = x + residual  # Residual connection
            features.append(x)
        
        # Multi-scale feature fusion
        x = torch.stack(features, dim=1).mean(dim=1)
        
        # Attention pooling
        att_weights = self.att_pool(x)
        weighted_sum = global_add_pool(x * att_weights, batch)
        max_pool = global_max_pool(x, batch)
        graph_rep = torch.cat([weighted_sum, max_pool], dim=1)
        
        # Classification
        return self.classifier(graph_rep)