import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, GATConv, global_mean_pool, global_add_pool
from torch.nn import Linear, BatchNorm1d, ReLU, Dropout

# class SimpleGCN(torch.nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(SimpleGCN, self).__init__()
#         self.embedding = torch.nn.Embedding(500, input_dim) 
#         # self.conv1 = GCNConv(input_dim, hidden_dim)
#         # self.conv2 = GCNConv(hidden_dim, hidden_dim)
#         self.global_pool = global_mean_pool  
#         self.fc = torch.nn.Linear(hidden_dim, output_dim)

#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#         x = self.embedding(x)  
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index)
#         x = self.global_pool(x, batch)  
#         out = self.fc(x)  
#         return out

class SimpleGCN(nn.Module):  # You could rename this to SimpleGIN if more accurate
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleGCN, self).__init__()

        # Optional: if you're using node indices as input
        self.embedding = nn.Embedding(500, input_dim)

        # Define the MLPs for each GIN layer
        self.conv1 = GINConv(nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ))

        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ))

        self.global_pool = global_mean_pool
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Assumes node indices as input to embedding
        x = self.embedding(x.view(-1).long())  # Ensure proper shape and dtype

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.global_pool(x, batch)
        out = self.fc(x)
        return out
    
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
        x = self.embedding(data.x.view(-1).long())

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
    
class paperGIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super().__init__()

        def mlp():
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )

        self.initial_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.conv1 = GINConv(mlp())
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.conv2 = GINConv(mlp())
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.conv3 = GINConv(mlp())
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.embedding = torch.nn.Embedding(500, input_dim) 

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.embedding(x)  

        x = self.initial_mlp(x)  # Project input features to hidden space

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)

        x = global_add_pool(x, batch)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class RobustGIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(RobustGIN, self).__init__()
        self.embedding = torch.nn.Embedding(500, input_dim)
        
        nn1 = torch.nn.Sequential(
            Linear(input_dim, hidden_dim),
            BatchNorm1d(hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU()
        )
        self.conv1 = GINConv(nn1)
        
        nn2 = torch.nn.Sequential(
            Linear(hidden_dim, hidden_dim),
            BatchNorm1d(hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU()
        )
        self.conv2 = GINConv(nn2)
        
        self.lin1 = Linear(hidden_dim * 2, hidden_dim)
        self.lin2 = Linear(hidden_dim, output_dim)
        self.dropout = Dropout(dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.embedding(x.squeeze().long())
        
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        
        h = torch.cat([h1, h2], dim=1)
        h = global_add_pool(h, batch)
        
        node_logits = x

        h = self.lin1(h)
        h = F.relu(h)
        h = self.dropout(h)
        out = self.lin2(h)

        return out, node_logits


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
        x, edge_index, batch = data.x, data.edge_index, data.batchs
        #x = self.embedding(x)  
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