import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, GATConv, global_mean_pool, global_add_pool, global_max_pool, GlobalAttention, Set2Set, GINEConv
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import uniform

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


# class SimpleStructLearner(nn.Module):
#     """
#     Learns a soft adjacency A_hat = sigmoid(Z Z^T) from node features,
#     with an L1 sparsity penalty on A_hat.
#     """
#     def __init__(self, feat_dim, hidden_dim, sparsity_coef=1e-3):
#         super().__init__()
#         # MLP encoder: x -> Z
#         self.encoder = nn.Sequential(
#             nn.Linear(feat_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim)
#         )
#         self.sparsity_coef = sparsity_coef

#     def forward(self, x, edge_index):
#         # x: [N, F]; edge_index: [2, E]
#         N = x.size(0)
#         Z = self.encoder(x)                # [N, H]
#         A_hat = torch.sigmoid(Z @ Z.t())   # [N, N] full soft-adj

#         # Extract only the edge weights we need:
#         src, dst = edge_index
#         edge_weight = A_hat[src, dst]      # [E]

#         # Register a sparsity loss that you can add to your total:
#         # (call model.struct_learner.loss() in your training loop)
#         self._sparsity_loss = self.sparsity_coef * A_hat.abs().mean()

#         return edge_weight

#     def loss(self):
#         # Must be added to your classification loss:
#         return self._sparsity_loss

class SimpleStructLearner(nn.Module):
    """
    For each edge, predicts a weight w_ij = sigmoid( MLP( [x_i || x_j] ) ).
    Also applies an L1‐style sparsity penalty on those E values.
    """
    def __init__(self, feat_dim, hidden_dim, sparsity_coef=1e-3):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.sparsity_coef = sparsity_coef

    def forward(self, x, edge_index):
        # x: [N, F], edge_index: [2, E]
        src, dst = edge_index
        h_edge = torch.cat([x[src], x[dst]], dim=1)  # [E, 2F]
        logits = self.edge_mlp(h_edge).squeeze()     # [E]
        w = torch.sigmoid(logits)                    # [E]

        # sparsity loss only on these E weights:
        self._sparsity_loss = self.sparsity_coef * w.abs().mean()
        return w

    def loss(self):
        return self._sparsity_loss


class SLGAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, num_heads=4, dropout=0.5, struct_hidden=64, sparsity_coef=1e-3):
        super().__init__()
        # 0) Embedding layer, exactly like your GIN
        self.embedding = nn.Embedding(500, input_dim)

        # 1) Structure-Learning module
        # self.struct_learner = SimpleStructLearner(
        #     feat_dim=input_dim,
        #     hidden_dim=struct_hidden,
        #     sparsity_coef=sparsity_coef
        # )
        self.struct_learner = SimpleStructLearner(
            feat_dim=input_dim,
            hidden_dim=struct_hidden,
            sparsity_coef=sparsity_coef
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
        x = self.embedding(x.long())   # [num_nodes, input_dim]

        # — 1) Learn per-edge weights & sparsity penalty —
        edge_weight = self.struct_learner(x, edge_index)

        # — 2) GAT on the denoised graph —
        h1 = F.elu(self.gat1(x, edge_index, edge_weight))
        h1 = self.dropout(h1)
        h2 = F.elu(self.gat2(h1, edge_index, edge_weight))
        h2 = self.dropout(h2)

        # — 3) JK readout —
        g1 = global_add_pool(h1, batch)
        g2 = global_add_pool(h2, batch)
        h  = torch.cat([g1, g2], dim=1)      # [batch_size, 2*hidden_dim]
        h  = F.relu(self.jump(h))
        h  = self.dropout(h)

        # — 4) Final MLP head —
        h  = F.relu(self.lin1(h)); h = self.dropout(h)
        out = self.lin2(h)

        return out

    def loss(self, classification_loss):
        """
        Combine your supervised loss with the sparsity loss:
            total_loss = classification_loss + struct_learner.loss()
        """
        return classification_loss + self.struct_learner.loss()
    

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

        x = global_mean_pool(x, batch)

        x = self.lin(x)

        return x