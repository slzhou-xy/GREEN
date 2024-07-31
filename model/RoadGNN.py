import torch.nn as nn
from torch_geometric.nn import GATConv
import torch.nn.functional as F



class RoadGNN(nn.Module):
    def __init__(self, fea_size, g_dim_per_layer, g_heads_per_layer, num_layers, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(fea_size, g_dim_per_layer[0])
        self.gat_net = nn.ModuleList([
            GATConv(in_channels=g_dim_per_layer[i],
                    out_channels=g_dim_per_layer[i] // g_heads_per_layer[i],
                    heads=g_heads_per_layer[i],
                    dropout=dropout
                    )
            for i in range(num_layers)
        ])

    def forward(self, x, edge_index):
        x = self.linear(x)
        for layer in self.gat_net[:-1]:
            x = F.relu(x + layer(x, edge_index), inplace=True)
        x = self.gat_net[-1](x, edge_index)
        return x