import copy

import torch
import torch_geometric
from sklearn.metrics import f1_score


def _macro_f1_and_acc(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
    pred = logits.argmax(dim=1)
    y_true = y[mask].cpu()
    y_pred = pred[mask].cpu()
    f1 = f1_score(y_true, y_pred, average='macro')
    acc = (y_pred == y_true).sum().item() / len(y_true)
    return f1, acc


def train_and_evaluate(
    model: torch.nn.Module,
    data: torch_geometric.data.Data,
    device: torch.device,
    class_weights: torch.Tensor,
    max_epochs: int,
    patience: int
):
    model = model.to(device)
    data = data.to(device)
    class_weights = class_weights.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    best_val_f1 = -1.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        logits = model(data.x, data.train_edge_index, data.train_edge_attr)
        loss_value = criterion(logits[data.train_mask], data.y[data.train_mask])
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_val = model(data.x, data.trainval_edge_index, data.trainval_edge_attr)
            val_f1, _ = _macro_f1_and_acc(logits_val, data.y, data.val_mask)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits_test = model(data.x, data.edge_index, data.edge_attr)
        test_f1, test_acc = _macro_f1_and_acc(logits_test, data.y, data.test_mask)

    return test_f1, test_acc, model


def train_mlp(
        model: torch.nn.Module,
        features: torch.Tensor,
        targets: torch.Tensor,
        train_mask: torch.Tensor,
        val_mask: torch.Tensor,
        test_mask: torch.Tensor,
        device: torch.device,
        class_weights: torch.Tensor,
        epochs=500,
):
    model = model.to(device)
    features = features.to(device)
    targets = targets.to(device)
    class_weights_gpu = class_weights.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights_gpu)

    best_val_f1 = -1.0
    best_state = None
    patience, counter = 30, 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(features)
        loss = criterion(out[train_mask], targets[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_f1, _ = _macro_f1_and_acc(out, targets, val_mask)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = copy.deepcopy(model.state_dict())
                counter = 0
            else:
                counter += 1

        if counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(features)
        test_f1, test_acc = _macro_f1_and_acc(out, targets, test_mask)

    return test_f1, test_acc, model
