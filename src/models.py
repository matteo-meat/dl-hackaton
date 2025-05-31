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