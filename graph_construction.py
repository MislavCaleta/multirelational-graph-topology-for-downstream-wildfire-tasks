import torch
from sklearn.neighbors import NearestNeighbors
import numpy as np
from torch_geometric.data import Data

EARTH_RADIUS_KM = 6371.0


def haversine_pairwise(coords_a: torch.Tensor, coords_b: torch.Tensor) -> torch.Tensor:
    """Great-circle distance (km) between matched rows of two [E, 2] lat/lon tensors in degrees."""
    lat1, lon1 = torch.deg2rad(coords_a[:, 0]), torch.deg2rad(coords_a[:, 1])
    lat2, lon2 = torch.deg2rad(coords_b[:, 0]), torch.deg2rad(coords_b[:, 1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    return 2.0 * EARTH_RADIUS_KM * torch.asin(torch.sqrt(torch.clamp(a, min=0.0, max=1.0)))


def create_knn_edge_index(
    pos_tensor: torch.Tensor,
    k: int,
    metric: str = "euclidean"
):
    pos_np = pos_tensor.cpu().numpy()

    if metric == "haversine":
        # sklearn's haversine metric expects [lat, lon] in radians
        pos_np = np.radians(pos_np)
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm="ball_tree", metric="haversine").fit(pos_np)
    else:
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm="auto").fit(pos_np)
    distances, indices = nbrs.kneighbors(pos_np)

    source_nodes = np.repeat(np.arange(pos_np.shape[0]), k)
    target_nodes = indices[:, 1:].flatten()
    edge_index = torch.tensor(np.stack(arrays=[source_nodes, target_nodes], axis=0), dtype=torch.long)

    return edge_index

def compute_edge_attributes(
    edge_index: torch.Tensor,
    pos: torch.Tensor,
    time_tensor: torch.Tensor,
    metric: str = "euclidean"
):
    row, col = edge_index

    if metric == "haversine":
        # pos must be raw lat/lon in degrees
        dist = haversine_pairwise(pos[row], pos[col])
    else:
        dist = torch.norm(pos[row] - pos[col], p=2, dim=1)
    # the connection importance is proportional to the inverse distance in pos (time or spatial)
    weight = 1.0 - (dist / (dist.max() + 1e-8))
    weight = weight.unsqueeze(dim=1)

    # we implement a temporal direction attribute
    # every node is now explicitly told if it's neighbor is in the future or in the past
    time_diff = time_tensor[col] - time_tensor[row]
    direction = torch.sign(time_diff) 

    edge_attr = torch.cat([weight, direction], dim=1)

    return edge_attr

def apply_causal_filter(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    pos_temporal: torch.Tensor
):
    t = pos_temporal.squeeze()
    src, tgt = edge_index
    keep = t[src] <= t[tgt]
    return edge_index[:, keep], edge_attr[keep, :]


def _build_group_anchor_times(pos_temporal: torch.Tensor, group_ids):
    n = pos_temporal.numel()
    pos_t = pos_temporal.view(-1).tolist()
    if group_ids is None:
        return pos_t

    earliest = {}
    for i, gid in enumerate(group_ids):
        if not gid:
            continue
        t = pos_t[i]
        if gid not in earliest or t < earliest[gid]:
            earliest[gid] = t

    anchor = [0.0] * n
    for i, gid in enumerate(group_ids):
        anchor[i] = earliest[gid] if gid else pos_t[i]
    return anchor


def get_split_edges_attr(
    pos_temporal: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    train_cutoff_percent: float,
    val_cutoff_percent: float,
    group_ids=None
):
    anchor = _build_group_anchor_times(pos_temporal, group_ids)
    anchor_t = torch.tensor(anchor, dtype=pos_temporal.dtype).view(-1)

    train_cutoff_value = torch.quantile(anchor_t, train_cutoff_percent)
    val_cutoff_value = torch.quantile(anchor_t, val_cutoff_percent)

    train_mask = anchor_t < train_cutoff_value
    val_mask = (anchor_t >= train_cutoff_value) & (anchor_t < val_cutoff_value)
    test_mask = anchor_t >= val_cutoff_value

    row, col = edge_index
    is_train_edge = train_mask[row] & train_mask[col]

    trainval_mask = train_mask | val_mask
    is_trainval_edge = trainval_mask[row] & trainval_mask[col]

    train_edge_index = edge_index[:, is_train_edge]
    train_edge_attr = edge_attr[is_train_edge, :]
    trainval_edge_index = edge_index[:, is_trainval_edge]
    trainval_edge_attr = edge_attr[is_trainval_edge, :]

    return (
        train_edge_index, train_edge_attr,
        trainval_edge_index, trainval_edge_attr,
        train_mask, val_mask, test_mask,
    )

def build_graph(
    graph_type: str,
    neighbors: int,
    pos_spatial: torch.Tensor,
    pos_temporal: torch.Tensor,
    pos_combined: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    causal: bool = True,
    group_ids=None,
    distance_metric: str = "euclidean",
    pos_spatial_raw: torch.Tensor = None
):
    edge_index = None
    edge_attr = None

    # under 'haversine' all spatial distances (kNN selection + edge weights) use
    # raw lat/lon in degrees; pos_spatial stays standardized for its node-feature role.
    if distance_metric == "haversine" and pos_spatial_raw is None:
        raise ValueError(
            "distance_metric='haversine' requires pos_spatial_raw (raw lat/lon in degrees); "
            "get it via prepare_dataset(..., return_raw_spatial=True)."
        )
    spatial_metric = distance_metric
    # geographic coords used for spatial distance computations
    pos_spatial_geo = pos_spatial_raw if spatial_metric == "haversine" else pos_spatial

    if graph_type == "spatial":
        edge_index = create_knn_edge_index(pos_spatial_geo, neighbors, metric=spatial_metric)
        edge_attr = compute_edge_attributes(edge_index, pos_spatial_geo, pos_temporal, metric=spatial_metric)
    elif graph_type == "temporal":
        edge_index = create_knn_edge_index(pos_temporal, neighbors)
        edge_attr = compute_edge_attributes(edge_index, pos_temporal, pos_temporal)
    elif graph_type == "combined":
        # kNN runs on the mixed (standardized) space, so it stays euclidean;
        # only the spatial edge-weight channel can use haversine.
        edge_index = create_knn_edge_index(pos_combined, neighbors)
        edge_attr_spatial = compute_edge_attributes(edge_index, pos_spatial_geo, pos_temporal, metric=spatial_metric)
        edge_attr_temporal = compute_edge_attributes(edge_index, pos_temporal, pos_temporal)
        edge_attr = torch.cat([edge_attr_spatial[:, 0:1], edge_attr_temporal[:, 0:1], edge_attr_spatial[:, 1:2]], dim=1)
    elif graph_type == "multiplex":
        k_s = neighbors // 2
        k_t = neighbors - k_s

        #avoid signal dilution that "combined" introduces
        edge_index_s = create_knn_edge_index(pos_spatial_geo, k_s, metric=spatial_metric)
        edge_index_t = create_knn_edge_index(pos_temporal, k_t)
        edge_index = torch.cat([edge_index_s, edge_index_t], dim=1)

        #we explicitly tell the model if this edge carries more important
        #temporal information or spatial information by assigning edge type
        type_s = torch.full((edge_index_s.size(1), 1), 0)
        type_t = torch.full((edge_index_t.size(1), 1), 1)
        edge_type = torch.cat([type_s, type_t], dim=0)
        edge_attr_spatial = compute_edge_attributes(edge_index, pos_spatial_geo, pos_temporal, metric=spatial_metric)
        edge_attr_temporal = compute_edge_attributes(edge_index, pos_temporal, pos_temporal)
        edge_attr = torch.cat(
            [
                edge_attr_spatial[:, 0:1],
                edge_attr_temporal[:, 0:1],
                edge_attr_spatial[:, 1:2],
                edge_type
            ],
            dim=1
        )

    if edge_index is None:
        raise ValueError(
            f"Invalid graph_type provided: {graph_type}. "
            f"Supported types: 'spatial', 'temporal', 'combined', 'multiplex'"
        )

    if causal:
        edge_index, edge_attr = apply_causal_filter(edge_index, edge_attr, pos_temporal)

    (
        train_edge_index, train_edge_attr,
        trainval_edge_index, trainval_edge_attr,
        train_mask, val_mask, test_mask,
    ) = get_split_edges_attr(
        pos_temporal,
        edge_index,
        edge_attr,
        train_cutoff_percent=0.72,
        val_cutoff_percent=0.80,
        group_ids=group_ids,
    )

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        train_edge_index=train_edge_index,
        train_edge_attr=train_edge_attr,
        trainval_edge_index=trainval_edge_index,
        trainval_edge_attr=trainval_edge_attr,
        y=y
    )
