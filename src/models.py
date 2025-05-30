import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, GATConv, global_mean_pool, global_add_pool, global_max_pool, GlobalAttention, Set2Set, GINEConv, VirtualNode
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import uniform
from torch_geometric.transforms import VirtualNode


from src.conv import GNN_node, GNN_node_Virtualnode

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
    

class GNN(torch.nn.Module):

    def __init__(self, num_class, num_layer = 5, emb_dim = 300, 
                    gnn_type = 'gin', virtual_node = True, residual = True, drop_ratio = 0.5, JK = "last", graph_pooling = "mean"):
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

        node_embeddings = x
        x = global_mean_pool(x, batch)

        x = self.lin(x)

        return x, node_embeddings
    
class EnhancedGINEWithVN(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim, drop_ratio=0.5):
        super().__init__()
        self.embedding = nn.Embedding(1, hidden_dim)
        self.conv1 = GINEConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ),
            edge_dim=7
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.conv2 = GINEConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ),
            edge_dim=7
        )
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # MLP to project pooled graph summary back to node space (“virtual node”)
        self.vn_lin = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.dropout = nn.Dropout(drop_ratio)
        self.lin = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index,
            data.edge_attr, data.batch
        )

        x0 = self.embedding(x)  # [total_nodes, hidden_dim]

        h1 = self.conv1(x0, edge_index, edge_attr)
        h1 = self.bn1(h1)
        h1 = F.relu(h1 + x0)    # skip connection from x0 → h1

        h2 = self.dropout(h1)
        h2 = self.conv2(h2, edge_index, edge_attr)
        h2 = self.bn2(h2)
        h2 = F.relu(h2 + h1)    # skip connection from h1 → h2

        # Virtual‐node via global pooling & broadcast —
        g = global_mean_pool(h2, batch)   # [num_graphs, hidden_dim]
        g_proj = self.vn_lin(g)                # [num_graphs, hidden_dim]
        g_exp = g_proj[batch]                 # [total_nodes, hidden_dim]
        h = h2 + g_exp                    # add global context

        graph_repr = global_mean_pool(h, batch) 
        out = self.lin(graph_repr) 

        return out, h