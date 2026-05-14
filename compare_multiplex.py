import statistics
import time
import numpy as np
import torch

from data_preparation import prepare_dataset, get_loss_weights
from graph_construction import build_graph
from models import GCN_Model, GAT_Model, Transformer_Model, RGCN_Model, BaselineMLP
from train_and_evaluate import train_and_evaluate, train_mlp
from resources import DATA_PATH


SEEDS = [42, 7, 1234, 2025, 88]
NEIGHBORS_LIST = [5, 7, 9]
HIDDEN_DIM = 64
MAX_EPOCHS = 200
PATIENCE = 30
MLP_EPOCHS = 500
SEASONAL_FEATURES = True


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_mlp_baseline(x, y, train_mask, val_mask, test_mask, weights, device):
    input_dim = x.size(-1)
    f1s, accs = [], []
    n_params = sum(p.numel() for p in BaselineMLP(input_dim, HIDDEN_DIM).parameters())
    t0 = time.time()
    for seed in SEEDS:
        set_seed(seed)
        model = BaselineMLP(input_dim=input_dim, hidden_dim=HIDDEN_DIM)
        f1, acc, _ = train_mlp(
            model=model,
            features=x,
            targets=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            device=device,
            class_weights=weights,
            epochs=MLP_EPOCHS,
        )
        f1s.append(f1)
        accs.append(acc)
    dt = time.time() - t0
    return {
        "f1_runs": f1s,
        "f1_mean": statistics.mean(f1s),
        "f1_std": statistics.stdev(f1s),
        "acc_mean": statistics.mean(accs),
        "acc_std": statistics.stdev(accs),
        "params": n_params,
        "total_time": dt,
    }


def run_one_k(k, x, y, pos_combined, pos_spatial, pos_temporal, group_ids, weights, device):
    data = build_graph(
        graph_type="multiplex",
        neighbors=k,
        pos_spatial=pos_spatial,
        pos_temporal=pos_temporal,
        pos_combined=pos_combined,
        x=x,
        y=y,
        group_ids=group_ids,
    )
    edge_dim = data.edge_attr.size(-1)
    input_dim = x.size(-1)
    n_train = data.train_mask.sum().item()
    n_val = data.val_mask.sum().item()
    n_test = data.test_mask.sum().item()
    print(f"\n--- k={k}  Edges: {data.num_edges}, edge_attr dim: {edge_dim} ---")
    print(f"     train: {n_train}, val: {n_val}, test: {n_test}")

    configs = [
        ("GCN", lambda: GCN_Model(input_dim=input_dim, hidden_dim=HIDDEN_DIM)),
        ("GAT", lambda: GAT_Model(input_dim=input_dim, hidden_dim=HIDDEN_DIM, edge_dim=edge_dim)),
        ("TransformerConv", lambda: Transformer_Model(input_dim=input_dim, hidden_dim=HIDDEN_DIM, edge_dim=edge_dim)),
        ("RGCN", lambda: RGCN_Model(input_dim=input_dim, hidden_dim=HIDDEN_DIM, num_relations=2)),
    ]

    summary = {}
    for name, factory in configs:
        f1s, accs = [], []
        n_params = sum(p.numel() for p in factory().parameters())
        t0 = time.time()
        for seed in SEEDS:
            set_seed(seed)
            model = factory()
            f1, acc, _ = train_and_evaluate(
                model=model,
                data=data,
                device=device,
                class_weights=weights,
                max_epochs=MAX_EPOCHS,
                patience=PATIENCE,
            )
            f1s.append(f1)
            accs.append(acc)
        dt = time.time() - t0
        summary[name] = {
            "f1_runs": f1s,
            "f1_mean": statistics.mean(f1s),
            "f1_std": statistics.stdev(f1s),
            "acc_mean": statistics.mean(accs),
            "acc_std": statistics.stdev(accs),
            "params": n_params,
            "total_time": dt,
        }
        print(f"  {name:<18s} F1 mean={summary[name]['f1_mean']:.4f} std={summary[name]['f1_std']:.4f}  ({dt:.1f}s)")

    return summary


def main():
    device = torch.device("cpu")

    set_seed(SEEDS[0])
    x, y, pos_combined, pos_spatial, pos_temporal, group_ids = prepare_dataset(
        DATA_PATH, seasonal_features=SEASONAL_FEATURES, return_group_ids=True
    )
    weights = get_loss_weights(y)
    print(f"\nNodes: {x.size(0)}  Node feature dim: {x.size(-1)}  Seeds: {SEEDS}  k values: {NEIGHBORS_LIST}")
    print(f"Seasonal features: {SEASONAL_FEATURES}")
    n_grouped_rows = sum(1 for g in group_ids if g)
    print(f"Rows with non-null mtbs_ID: {n_grouped_rows}")

    all_summaries = {}
    for k in NEIGHBORS_LIST:
        all_summaries[k] = run_one_k(
            k, x, y, pos_combined, pos_spatial, pos_temporal, group_ids, weights, device
        )

    ref_data = build_graph(
        graph_type="multiplex",
        neighbors=NEIGHBORS_LIST[0],
        pos_spatial=pos_spatial,
        pos_temporal=pos_temporal,
        pos_combined=pos_combined,
        x=x,
        y=y,
        group_ids=group_ids,
    )
    print("\n--- BaselineMLP (no graph, same masks) ---")
    mlp_summary = run_mlp_baseline(
        x, y, ref_data.train_mask, ref_data.val_mask, ref_data.test_mask, weights, device
    )
    print(f"  BaselineMLP        F1 mean={mlp_summary['f1_mean']:.4f} std={mlp_summary['f1_std']:.4f}  ({mlp_summary['total_time']:.1f}s)")

    print("\n\n=== Final summary: multiplex, all k, mean ± std over {} seeds ===".format(len(SEEDS)))
    print(f"{'Model':<18s}", end="")
    for k in NEIGHBORS_LIST:
        print(f"  k={k:<2d} F1 mean ± std    ", end="")
    print()
    for name in ["GCN", "GAT", "TransformerConv", "RGCN"]:
        print(f"{name:<18s}", end="")
        for k in NEIGHBORS_LIST:
            s = all_summaries[k][name]
            print(f"   {s['f1_mean']:.4f} ± {s['f1_std']:.4f}    ", end="")
        print()
    print(f"\n{'BaselineMLP (no graph)':<24s} F1 = {mlp_summary['f1_mean']:.4f} ± {mlp_summary['f1_std']:.4f}  Acc = {mlp_summary['acc_mean']:.4f} ± {mlp_summary['acc_std']:.4f}")


if __name__ == "__main__":
    main()
