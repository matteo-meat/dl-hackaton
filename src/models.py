import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform
from torch_geometric.nn import GCNConv, GINConv, GINEConv, GATConv, global_mean_pool
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set

from src.conv import GNN_node, GNN_node_Virtualnode

class SimpleGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleGCN, self).__init__()
        self.embedding = torch.nn.Embedding(1, input_dim) 
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
        self.embedding = torch.nn.Embedding(1, input_dim) 
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
    
class GNN(torch.nn.Module):

    def __init__(self, num_class, num_layer = 5, emb_dim = 300, 
                    gnn_type = 'gin', virtual_node = True, residual = False, drop_ratio = 0.5, JK = "last", graph_pooling = "mean"):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_class = num_class
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)
        else:
            self.gnn_node = GNN_node(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)


        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim, self.num_class)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_class)

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        h_graph = self.pool(h_node, batched_data.batch)

        return self.graph_pred_linear(h_graph)
    
class SimpleGIN(torch.nn.Module):

    def __init__(self, hidden_dim, output_dim, drop_ratio = 0.5):
        super(SimpleGIN, self).__init__()

        self.drop_ratio = drop_ratio

        self.node_embedding = nn.Embedding(1, hidden_dim)

        nn1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), 
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim))
        self.conv1 = GINConv(nn1, edge_dim = 7)

        nn2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), 
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim))
        
        self.conv2 = GINConv(nn2, edge_dim = 7)

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
    
class SimpleGINE(torch.nn.Module):

    def __init__(self, hidden_dim, output_dim, drop_ratio = 0.5):
        super(SimpleGINE, self).__init__()

        self.drop_ratio = drop_ratio

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
    
class GINEPaper(torch.nn.Module):

    def __init__(self, hidden_dim, output_dim, drop_ratio = 0.5):
        super(SimpleGINE, self).__init__()

        self.drop_ratio = drop_ratio

        self.node_embedding = nn.Embedding(1, hidden_dim)

        nn1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), 
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim))
        self.conv1 = GINEConv(nn1, edge_dim = 7)

        nn2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), 
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim))
        
        self.conv2 = GINEConv(nn2, edge_dim = 7)

        nn3 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), 
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim))
        self.conv3 = GINEConv(nn3, edge_dim = 7)

        nn4 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), 
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim))
        
        self.conv4 = GINEConv(nn4, edge_dim = 7)

        nn5 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), 
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim))
        
        self.conv5 = GINEConv(nn5, edge_dim = 7)

        self.lin = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.node_embedding(x)

        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv4(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv5(x, edge_index, edge_attr)

        x = global_mean_pool(x, batch)

        x = self.lin(x)

        return x
