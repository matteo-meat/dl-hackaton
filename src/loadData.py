import gzip
import json
import torch
from torch_geometric.data import Dataset, Data
from tqdm import tqdm 

class GraphDataset(Dataset):
    def __init__(self, filename, transform=None, pre_transform=None):
        self.raw = filename
        self.graphs = self.loadGraphs(self.raw)
        super().__init__(None, transform, pre_transform)
        self.original_indices = list(range(len(self.graphs)))

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        data = self.graphs[idx]
        data.idx = self.original_indices[idx]
        return data

    @staticmethod
    def loadGraphs(path):
        print(f"Loading graphs from {path}...")
        print("This may take a few minutes, please wait...")
        with gzip.open(path, "rt", encoding="utf-8") as f:
            graphs_dicts = json.load(f)
        graphs = []
        for i, graph_dict in enumerate(tqdm(graphs_dicts, desc="Processing graphs", unit="graph")):
            data = dictToGraphObject(graph_dict)
            data.idx = i
            graphs.append(data)
        return graphs



def dictToGraphObject(graph_dict):
    edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long)
    edge_attr = torch.tensor(graph_dict["edge_attr"], dtype=torch.float) if graph_dict["edge_attr"] else None
    num_nodes = graph_dict["num_nodes"]
    y = torch.tensor(graph_dict["y"][0], dtype=torch.long) if graph_dict["y"] is not None else None
    return Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes, y=y)








