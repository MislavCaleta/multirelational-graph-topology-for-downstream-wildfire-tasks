import torch
from torch_geometric.nn import (
    GCNConv,
    GATv2Conv,
    TransformerConv,
    RGCNConv,
    Linear
)

class GCN_Model(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels=input_dim, out_channels=hidden_dim)
        self.conv2 = GCNConv(in_channels=hidden_dim, out_channels=hidden_dim)
        self.classifier = Linear(in_channels=hidden_dim, out_channels=2)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None):
        if edge_attr.size(-1) > 2:
            edge_weight = edge_attr[:, 0:2].mean(dim=-1)
        else:
            edge_weight = edge_attr[:, 0]

        x = self.conv1(x, edge_index, edge_weight).relu()
        x = torch.nn.functional.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index, edge_weight).relu()
        x = torch.nn.functional.dropout(x, p=0.2, training=self.training)
        x = self.classifier(x)

        return x

class GAT_Model(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, edge_dim: int):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels=input_dim, out_channels=hidden_dim, edge_dim=edge_dim)
        self.conv2 = GATv2Conv(in_channels=hidden_dim, out_channels=hidden_dim, edge_dim=edge_dim)
        self.classifier = Linear(in_channels=hidden_dim, out_channels=2)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None):
        x = self.conv1(x, edge_index, edge_attr).relu()
        x = torch.nn.functional.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index, edge_attr).relu()
        x = torch.nn.functional.dropout(x, p=0.2, training=self.training)
        x = self.classifier(x)

        return x

class Transformer_Model(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, edge_dim: int):
        super().__init__()
        self.conv1 = TransformerConv(in_channels=input_dim, out_channels=hidden_dim, edge_dim=edge_dim)
        self.conv2 = TransformerConv(in_channels=hidden_dim, out_channels=hidden_dim, edge_dim=edge_dim)
        self.classifier = Linear(in_channels=hidden_dim, out_channels=2)

    def forward(self, x:torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None):
        x = self.conv1(x, edge_index, edge_attr).relu()
        x = torch.nn.functional.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index, edge_attr).relu()
        x = torch.nn.functional.dropout(x, p=0.2, training=self.training)
        x = self.classifier(x)

        return x

class BaselineMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(BaselineMLP, self).__init__()
        self.layer1 = torch.nn.Linear(input_dim, hidden_dim)
        self.layer2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.classifier = torch.nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.layer1(x).relu()
        x = torch.nn.functional.dropout(x, p=0.2, training=self.training)
        x = self.layer2(x).relu()
        x = torch.nn.functional.dropout(x, p=0.2, training=self.training)
        return self.classifier(x)
    
class GoldenTransformer(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, edge_dim, num_heads):
        super().__init__()
        self.conv1 = TransformerConv(input_dim, hidden_dim, edge_dim=edge_dim, heads=num_heads, dropout=0.2)
        self.norm1 = torch.nn.LayerNorm(hidden_dim * num_heads)
        self.conv2 = TransformerConv(hidden_dim * num_heads, hidden_dim, edge_dim=edge_dim, heads=num_heads, concat=False)
        self.classifier = torch.nn.Linear(hidden_dim, 2)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index, edge_attr).relu()
        x = self.norm1(x)
        x = torch.nn.functional.dropout(x, p=0.2, training=self.training)

        x = self.conv2(x, edge_index, edge_attr).relu()
        x = torch.nn.functional.dropout(x, p=0.2, training=self.training)

        return self.classifier(x)

class RGCN_Model(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_relations: int = 2):
        super().__init__()
        self.conv1 = RGCNConv(in_channels=input_dim, out_channels=hidden_dim, num_relations=num_relations)
        self.conv2 = RGCNConv(in_channels=hidden_dim, out_channels=hidden_dim, num_relations=num_relations)
        self.classifier = Linear(in_channels=hidden_dim, out_channels=2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None):
        edge_type = edge_attr[:, -1].long()
        x = self.conv1(x, edge_index, edge_type).relu()
        x = torch.nn.functional.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index, edge_type).relu()
        x = torch.nn.functional.dropout(x, p=0.2, training=self.training)
        return self.classifier(x)