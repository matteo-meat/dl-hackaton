import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, GATConv, global_mean_pool, global_add_pool
from torch_geometric.nn.models import ProGNN

class SimpleGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleGCN, self).__init__()
        self.embedding = torch.nn.Embedding(500, input_dim) 
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        # self.conv1 = GINConv(torch.nn.Linear(input_dim, hidden_dim), torch.nn.ReLU(), torch.nn.Linear(hidden_dim, hidden_dim))
        # self.conv2 = GINConv(torch.nn.Linear(input_dim, hidden_dim), torch.nn.ReLU(), torch.nn.Linear(hidden_dim, hidden_dim))
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
    
class CulturalClassificationGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, use_gat = False):
        super(CulturalClassificationGNN, self).__init__()
        
        self.use_gat = use_gat
        # Graph convolutional layers
        self.embedding = torch.nn.Embedding(500, input_dim) 
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        if self.use_gat:
            self.gat = GATConv(hidden_dim, hidden_dim, heads=4, dropout=0.3)
            self.fc1 = nn.Linear(hidden_dim * 4, hidden_dim)
        else:
            self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.embedding(x)  
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))

        if self.use_gat:
            x = self.gat(x, edge_index)

        x = global_mean_pool(x, batch)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)
    

class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5 ):
        super().__init__()
        self.embedding = torch.nn.Embedding(500, input_dim) 
        self.conv1 = GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        self.conv2 = GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        self.conv3 = GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        self.lin1 = nn.Linear(hidden_dim*3, hidden_dim*3)
        self.lin2 = nn.Linear(hidden_dim*3, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.embedding(x)

        # Node embeddings - h1,h2,h3 are node level features
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        # Compressing all nodde features in a graph into a single vector per graph
        g1 = global_add_pool(h1, batch)
        g2 = global_add_pool(h2, batch)
        g3 = global_add_pool(h3, batch)

        # Concatenate graph embeddings
        h = torch.cat((g1,g2,g3), dim=1)

        # Classification head
        h = self.lin1(h)
        h = F.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)
        return h
    

class SLGAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, num_heads=5, dropout=0.5):
        super().__init__()
        # 0) Embedding layer, exactly like your GIN
        self.embedding = nn.Embedding(500, input_dim)

        # 1) Structure-Learning module
        self.struct_learner = ProGNN(
            in_feats=input_dim,
            hidden_dims=[hidden_dim],
            alpha=0.1,
            beta=0.1,
            learn_eps=True
        )

        # 2) GAT layers: note in_feats=input_dim for the first layer
        self.gat1 = GATConv(input_dim,
                            hidden_dim // num_heads,
                            heads=num_heads,
                            dropout=dropout)
        # second layer now takes hidden_dim in → hidden_dim//heads
        self.gat2 = GATConv(hidden_dim,
                            hidden_dim // num_heads,
                            heads=num_heads,
                            dropout=dropout)

        # 3) Jumping-Knowledge projection
        self.jump = nn.Linear(2 * hidden_dim, hidden_dim)

        # 4) MLP head
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # — Embedding —
        x = self.embedding(x.long())   # [num_nodes, input_dim]

        # — Structure learning —
        new_edge_index, edge_weight = self.struct_learner(x, edge_index)

        # — GAT convs —
        h1 = self.gat1(x, new_edge_index, edge_weight)
        h1 = F.elu(h1); h1 = self.dropout(h1)

        h2 = self.gat2(h1, new_edge_index, edge_weight)
        h2 = F.elu(h2); h2 = self.dropout(h2)

        # — JK readout —
        g1 = global_add_pool(h1, batch)
        g2 = global_add_pool(h2, batch)
        h  = torch.cat([g1, g2], dim=1)        # size = [batch_size, 2*hidden_dim]
        h  = F.relu(self.jump(h))              # project back to hidden_dim
        h  = self.dropout(h)

        # — MLP head —
        h  = F.relu(self.lin1(h)); h = self.dropout(h)
        return self.lin2(h)
