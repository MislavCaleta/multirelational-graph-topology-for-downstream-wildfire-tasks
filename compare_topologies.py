import csv
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from data_preparation import prepare_dataset, get_loss_weights
from graph_construction import build_graph
from models import GCN_Model, GAT_Model, Transformer_Model, RGCN_Model, BaselineMLP
from train_and_evaluate import train_and_evaluate, train_mlp
from resources import DATA_PATH


SEEDS = [42, 7, 1234, 2025, 88]
NEIGHBORS_LIST = [5, 7, 9]
TOPOLOGIES = ["spatial", "temporal", "combined", "multiplex"]
HIDDEN_DIM = 64
MAX_EPOCHS = 200
PATIENCE = 30
MLP_EPOCHS = 500
SEASONAL_FEATURES = True
OUTPUT_CSV = Path("outputs/tables/clean_eval_topology_sweep.csv")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gnn_configs(topology, input_dim, edge_dim):
    base = [
        ("GCN", lambda: GCN_Model(input_dim=input_dim, hidden_dim=HIDDEN_DIM)),
        ("GAT", lambda: GAT_Model(input_dim=input_dim, hidden_dim=HIDDEN_DIM, edge_dim=edge_dim)),
        ("TransformerConv", lambda: Transformer_Model(input_dim=input_dim, hidden_dim=HIDDEN_DIM, edge_dim=edge_dim)),
    ]
    if topology == "multiplex":
        base.append(("RGCN", lambda: RGCN_Model(input_dim=input_dim, hidden_dim=HIDDEN_DIM, num_relations=2)))
    return base


def run_models(data, configs, weights, device):
    out = {}
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
        out[name] = {
            "f1_mean": statistics.mean(f1s),
            "f1_std": statistics.stdev(f1s) if len(f1s) > 1 else 0.0,
            "acc_mean": statistics.mean(accs),
            "acc_std": statistics.stdev(accs) if len(accs) > 1 else 0.0,
            "params": n_params,
            "time": dt,
        }
        print(f"    {name:<18s} F1={out[name]['f1_mean']:.4f} ± {out[name]['f1_std']:.4f}  ({dt:.1f}s)")
    return out


def run_mlp(x, y, train_mask, val_mask, test_mask, weights, device):
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
        "f1_mean": statistics.mean(f1s),
        "f1_std": statistics.stdev(f1s) if len(f1s) > 1 else 0.0,
        "acc_mean": statistics.mean(accs),
        "acc_std": statistics.stdev(accs) if len(accs) > 1 else 0.0,
        "params": n_params,
        "time": dt,
    }


def main():
    device = torch.device("cpu")

    set_seed(SEEDS[0])
    x, y, pos_combined, pos_spatial, pos_temporal, group_ids = prepare_dataset(
        DATA_PATH, seasonal_features=SEASONAL_FEATURES, return_group_ids=True
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
    weights = get_loss_weights(y, mask=ref_data.train_mask)

    n_train = ref_data.train_mask.sum().item()
    n_val = ref_data.val_mask.sum().item()
    n_test = ref_data.test_mask.sum().item()
    print(f"\nNodes: {x.size(0)}  Feature dim: {x.size(-1)}  Seasonal: {SEASONAL_FEATURES}")
    print(f"Split (group-aware on mtbs_ID): train={n_train}, val={n_val}, test={n_test}")
    print(f"Seeds: {SEEDS}, k values: {NEIGHBORS_LIST}, topologies: {TOPOLOGIES}\n")

    results = {}
    overall_t0 = time.time()
    for topology in TOPOLOGIES:
        results[topology] = {}
        for k in NEIGHBORS_LIST:
            data = build_graph(
                graph_type=topology,
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
            configs = gnn_configs(topology, input_dim, edge_dim)
            print(f"\n--- topology={topology}, k={k}, edges={data.num_edges}, edge_attr_dim={edge_dim} ---")
            results[topology][k] = run_models(data, configs, weights, device)
        print(f"\n  -- topology={topology} done in {(time.time() - overall_t0):.1f}s elapsed --")

    print("\n--- BaselineMLP (no graph) ---")
    mlp_summary = run_mlp(
        x, y, ref_data.train_mask, ref_data.val_mask, ref_data.test_mask, weights, device
    )
    print(f"    BaselineMLP        F1={mlp_summary['f1_mean']:.4f} ± {mlp_summary['f1_std']:.4f}  ({mlp_summary['time']:.1f}s)")

    print(f"\n\n=== Final summary: F1 macro ± std, {len(SEEDS)} seeds ===")
    print(f"{'Topology':<12s} {'Model':<18s}", end="")
    for k in NEIGHBORS_LIST:
        print(f"  k={k:<2d} F1 mean ± std    ", end="")
    print()
    for topology in TOPOLOGIES:
        for model_name in ["GCN", "GAT", "TransformerConv", "RGCN"]:
            row_data = []
            present = False
            for k in NEIGHBORS_LIST:
                if model_name in results[topology][k]:
                    s = results[topology][k][model_name]
                    row_data.append(f"   {s['f1_mean']:.4f} ± {s['f1_std']:.4f}    ")
                    present = True
                else:
                    row_data.append("        ---           ")
            if present:
                print(f"{topology:<12s} {model_name:<18s}", end="")
                for cell in row_data:
                    print(cell, end="")
                print()
    print(f"\n{'BaselineMLP (no graph)':<32s} F1 = {mlp_summary['f1_mean']:.4f} ± {mlp_summary['f1_std']:.4f}  Acc = {mlp_summary['acc_mean']:.4f} ± {mlp_summary['acc_std']:.4f}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["topology", "k", "model", "f1_mean", "f1_std", "acc_mean", "acc_std", "params", "time_seconds"])
        for topology in TOPOLOGIES:
            for k in NEIGHBORS_LIST:
                for model_name, s in results[topology][k].items():
                    writer.writerow([topology, k, model_name, f"{s['f1_mean']:.6f}", f"{s['f1_std']:.6f}",
                                     f"{s['acc_mean']:.6f}", f"{s['acc_std']:.6f}", s["params"], f"{s['time']:.2f}"])
        writer.writerow(["", "", "BaselineMLP", f"{mlp_summary['f1_mean']:.6f}", f"{mlp_summary['f1_std']:.6f}",
                         f"{mlp_summary['acc_mean']:.6f}", f"{mlp_summary['acc_std']:.6f}", mlp_summary["params"], f"{mlp_summary['time']:.2f}"])
    print(f"\nResults written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
