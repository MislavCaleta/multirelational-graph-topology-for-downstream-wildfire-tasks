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

    def extract_embeddings(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index, edge_attr).relu()
        x = self.norm1(x)
        x = self.conv2(x, edge_index, edge_attr).relu()
        return x

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


class HybridTransformerRF:
    """
    Hybrid Architecture combining:
    1) Multi-relational GoldenTransformer (spatiotemporal graph attention & message passing)
    2) RandomForestClassifier (non-linear tabular feature partitioning)

    Modes:
      - 'blend': Soft-voting ensemble of calibrated GNN and RF predictive probabilities.
      - 'stacking': RF trained on concatenation of raw features, GNN structural embeddings, and GNN logits.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        edge_dim: int,
        num_heads: int,
        rf_n_estimators: int = 100,
        rf_class_weight: str = 'balanced',
        mode: str = 'blend',
        gnn_weight: float = 0.5,
        random_state: int = 42,
    ):
        import numpy as np
        self.gnn = GoldenTransformer(input_dim, hidden_dim, edge_dim, num_heads)
        self.rf = None
        self.rf_n_estimators = rf_n_estimators
        self.rf_class_weight = rf_class_weight
        self.mode = mode
        self.gnn_weight = gnn_weight
        self.random_state = random_state

    def fit_gnn(self, data, device, class_weights, max_epochs=500, patience=30):
        from train_and_evaluate import train_and_evaluate
        _, _, self.gnn = train_and_evaluate(
            self.gnn, data, device, class_weights, max_epochs=max_epochs, patience=patience
        )

    def fit_rf(self, data, rf_features=None):
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier

        self.rf = RandomForestClassifier(
            n_estimators=self.rf_n_estimators,
            class_weight=self.rf_class_weight,
            random_state=self.random_state,
            n_jobs=-1
        )
        train_idx = data.train_mask.detach().cpu().numpy()
        y_np = data.y.detach().cpu().numpy()

        if self.mode == 'blend':
            X_tr = rf_features[train_idx] if rf_features is not None else data.x.detach().cpu().numpy()[train_idx]
            self.rf.fit(X_tr, y_np[train_idx])
        elif self.mode == 'stacking':
            hybrid_features = self._get_stacking_features(data, rf_features)
            self.rf.fit(hybrid_features[train_idx], y_np[train_idx])

    def _get_stacking_features(self, data, rf_features=None):
        import numpy as np
        self.gnn.eval()
        device = next(self.gnn.parameters()).device
        data = data.to(device)
        with torch.no_grad():
            embeds = self.gnn.extract_embeddings(data.x, data.edge_index, data.edge_attr)
            logits = self.gnn.classifier(embeds)
            probs = torch.softmax(logits, dim=1)[:, 1:2]
        
        base_x = rf_features if rf_features is not None else data.x.detach().cpu().numpy()
        return np.concatenate([base_x, embeds.cpu().numpy(), probs.cpu().numpy()], axis=1)

    def predict_proba(self, data, mask=None, rf_features=None):
        import numpy as np
        self.gnn.eval()
        device = next(self.gnn.parameters()).device
        data = data.to(device)
        with torch.no_grad():
            logits = self.gnn(data.x, data.edge_index, data.edge_attr)
            probs_gnn = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

        if self.mode == 'blend':
            X_eval = rf_features if rf_features is not None else data.x.detach().cpu().numpy()
            probs_rf = self.rf.predict_proba(X_eval)[:, 1]
            probs_hybrid = self.gnn_weight * probs_gnn + (1.0 - self.gnn_weight) * probs_rf
        elif self.mode == 'stacking':
            hybrid_x = self._get_stacking_features(data, rf_features)
            probs_hybrid = self.rf.predict_proba(hybrid_x)[:, 1]

        if mask is not None:
            m_np = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
            probs_hybrid = probs_hybrid[m_np]

        return np.column_stack([1.0 - probs_hybrid, probs_hybrid])

    def predict(self, data, mask=None, rf_features=None):
        probs = self.predict_proba(data, mask=mask, rf_features=rf_features)[:, 1]
        return (probs >= 0.5).astype(int)