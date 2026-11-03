"""The BA3Motif classifier ported from myexplainer's gnns/ba3motif_gnn.py."""

import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import LEConv, global_mean_pool


class BA3MotifLEConv(nn.Module):
    """Graph classifier with the source model's LEConv layers and readout."""

    def __init__(self, in_channels=10, num_unit=2, hidden_dim=64, dropout=0.4):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_classes = 3
        self.num_unit = num_unit
        self.dropout = dropout
        self.node_emb = nn.Linear(in_channels, hidden_dim)
        self.convs = nn.ModuleList(
            LEConv(hidden_dim, hidden_dim) for _ in range(num_unit)
        )
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, self.num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        self.node_emb.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def get_node_reps(self, x, edge_index, edge_weight=None):
        h = F.dropout(self.node_emb(x), p=self.dropout, training=self.training)
        for conv in self.convs:
            h = F.relu(conv(h, edge_index, edge_weight=edge_weight))
        return F.dropout(h, p=self.dropout, training=self.training)

    def get_graph_rep(self, x, edge_index, batch, edge_weight=None):
        return global_mean_pool(self.get_node_reps(x, edge_index, edge_weight), batch)

    def forward(self, x, edge_index, batch, edge_weight=None):
        h = self.get_graph_rep(x, edge_index, batch, edge_weight)
        return self.lin2(F.relu(self.lin1(h)))

    def get_pred(self, x, edge_index, batch):
        logits = self(x, edge_index, batch)
        return logits.softmax(dim=-1), logits

    def get_pred_explain(self, x, edge_index, edge_weight, batch):
        logits = self(x, edge_index, batch, edge_weight)
        return logits.softmax(dim=-1), logits
