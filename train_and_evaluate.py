import torch
import torch_geometric
from sklearn.metrics import f1_score

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

    best_f1 = 0
    best_acc = 0
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        # train using only the train edge index and train edge attributes
        model.train()
        logits = model(data.x, data.train_edge_index, data.train_edge_attr)
        loss_value = criterion(logits[data.train_mask], data.y[data.train_mask])
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        # during evaluation we use full edge index and full edge attributtes
        # we calculate loss only for test mask
        model.eval()
        with torch.no_grad():
            logits_val = model(data.x, data.edge_index, data.edge_attr)
            loss_value_val = criterion(logits[data.test_mask], data.y[data.test_mask])
        
            pred_val = logits_val.argmax(dim=1)

            y_true = data.y[data.test_mask].cpu()
            y_pred = pred_val[data.test_mask].cpu()

            f1 = f1_score(y_true, y_pred, average='macro')
            correct = (y_pred == y_true).sum().item()
            acc = correct / len(y_true)

            if f1 > best_f1:
                best_f1, best_acc = f1, acc
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            break
    
    return best_f1, best_acc, model

def train_mlp(
        model: torch.nn.Module,
        features: torch.Tensor,
        targets: torch.Tensor,
        t_mask: torch.Tensor,
        v_mask: torch.Tensor,
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

    best_f1, best_acc = 0, 0
    patience, counter = 30, 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(features)
        loss = criterion(out[t_mask], targets[t_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = out.argmax(dim=1)
            y_true = targets[v_mask].cpu()
            y_pred = pred[v_mask].cpu()
            
            f1 = f1_score(y_true, y_pred, average='macro')
            acc = (y_pred == y_true).sum().item() / len(y_true)

            if f1 > best_f1:
                best_f1, best_acc = f1, acc
                counter = 0
            else:
                counter += 1
        
        if counter >= patience: break

    return best_f1, best_acc, model