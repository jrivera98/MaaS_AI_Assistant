#graph_builder 

import torch
import pandas as pd
from torch_geometric.data import Data
import torch.nn.functional as F
from collections import deque,defaultdict


def build_pyg_graph(metro_edges_df, station_features_df, make_undirected=True):
    station_features_df = station_features_df.drop_duplicates(subset="station_id").reset_index(drop=True)

    station_ids = station_features_df["station_id"].tolist()
    station_to_idx = {sid: i for i, sid in enumerate(station_ids)}

    node_feature_cols = [
        col for col in station_features_df.columns
        if col != "station_id" and pd.api.types.is_numeric_dtype(station_features_df[col])
    ]

    x = torch.tensor(
        station_features_df[node_feature_cols].values,
        dtype=torch.float
    )

    valid_edges = metro_edges_df[
        metro_edges_df["station_id_from"].isin(station_to_idx) &
        metro_edges_df["station_id_to"].isin(station_to_idx)
    ].copy()

    src = valid_edges["station_id_from"].map(station_to_idx).tolist()
    dst = valid_edges["station_id_to"].map(station_to_idx).tolist()

    edge_pairs = list(zip(src, dst))

    if make_undirected:
        edge_pairs += [(b, a) for a, b in edge_pairs]

    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()

    edge_feature_cols = [
        col for col in metro_edges_df.columns
        if col not in ["station_id_from", "station_id_to"]
        and pd.api.types.is_numeric_dtype(metro_edges_df[col])
    ]

    edge_attr = None
    if edge_feature_cols:
        edge_values = valid_edges[edge_feature_cols].values

        if make_undirected:
            edge_values = pd.concat(
                [pd.DataFrame(edge_values), pd.DataFrame(edge_values)],
                ignore_index=True
            ).values

        edge_attr = torch.tensor(edge_values, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)

    if edge_attr is not None:
        data.edge_attr = edge_attr

    data.station_ids = station_ids
    data.station_to_idx = station_to_idx
    data.node_feature_names = node_feature_cols
    data.edge_feature_names = edge_feature_cols

    return data

# inference 

def load_trained_model(model_class, checkpoint_path, model_kwargs, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = model_class(**model_kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    metadata = {
        "node_feature_names": checkpoint["node_feature_names"],
        "edge_feature_names": checkpoint["edge_feature_names"],
        "station_ids": checkpoint["station_ids"],
    }

    return model, metadata

def run_multitask_inference(model, pyg_data):
    model.eval()
    with torch.no_grad():
        congestion_logits, feeder_logits = model(
            pyg_data.x,
            pyg_data.edge_index,
            pyg_data.edge_attr
        )

        congestion_probs = F.softmax(congestion_logits, dim=1)
        feeder_probs = F.softmax(feeder_logits, dim=1)

        congestion_pred = congestion_probs.argmax(dim=1)
        feeder_pred = feeder_probs.argmax(dim=1)

    predictions_df = pd.DataFrame({
        "station_id": pyg_data.station_ids,
        "pred_congestion_class": congestion_pred.cpu().numpy(),
        "pred_feeder_class": feeder_pred.cpu().numpy(),
        "pred_congestion_confidence": congestion_probs.max(dim=1).values.cpu().numpy(),
        "pred_feeder_confidence": feeder_probs.max(dim=1).values.cpu().numpy(),
    })

    return predictions_df, congestion_probs, feeder_probs

def add_readable_labels(dataframe):
    label_df = dataframe.copy()
    label_df["congestion_label"] = dataframe["pred_congestion_class"].map(CONGESTION_LABELS)
    label_df["feeder_label"] = dataframe["pred_feeder_class"].map(FEEDER_LABELS)
    return label_df

#routing 
def build_adjacency_list(metro_edges_df):
    adjacency_station = defaultdict(list)

    for _, row in metro_edges_df.iterrows():
        a = row["station_id_from"]
        b = row["station_id_to"]

        adjacency_station[a].append(b)
        adjacency_station[b].append(a)

    return adjacency_station

def generate_routes(adjacency, origin_station, destination_station, max_transfer, max_routes=5):
    routes = []
    queue = deque([[origin_station]])

    while queue and len(routes) < max_routes:
        path = queue.popleft()
        current = path[-1]

        if current == destination_station:
            routes.append(path)
            continue

        if len(path) > max_transfer:
            continue

        for neighbor in adjacency[current]:
            if neighbor not in path:
                new_path = path + [neighbor]
                queue.append(new_path)
    return routes

def score_routes(possible_routes, predictions_df):
    pred_map = predictions_df.set_index("station_id").to_dict("index")

    scored_routes = []

    for route in possible_routes:
        total_congestion_penalty = 0.0
        total_feeder_bonus = 0.0

        station_details = []

        for station_id in route:
            station_pred = pred_map.get(station_id, None)

            if station_pred is None:
                continue

            c_class = station_pred["pred_congestion_class"]
            f_class = station_pred["pred_feeder_class"]

            total_congestion_penalty += congestion_penalty(c_class)
            total_feeder_bonus += feeder_bonus(f_class)

            station_details.append({
                "station_id": station_id,
                "congestion_label": station_pred["congestion_label"],
                "feeder_label": station_pred["feeder_label"],
                "congestion_confidence": station_pred["pred_congestion_confidence"],
                "feeder_confidence": station_pred["pred_feeder_confidence"],
            })

        route_length_penalty = len(route) - 1

        final_score = route_length_penalty + total_congestion_penalty - total_feeder_bonus

        scored_routes.append({
            "route": route,
            "num_stops": len(route),
            "route_length_penalty": route_length_penalty,
            "total_congestion_penalty": total_congestion_penalty,
            "total_feeder_bonus": total_feeder_bonus,
            "final_score": final_score,
            "station_details": station_details
        })

    scored_routes.sort(key=lambda r: r["final_score"])
    return scored_routes

def recommend_routes_v1(origin_station,destination_station,metro_edges_df,station_features_df,model, max_transfer):
    # Build graph
    pyg_data = build_pyg_graph(metro_edges_df, station_features_df)

    #Run GAT predictions
    predictions_df, _, _ = run_multitask_inference(model, pyg_data)
    predictions_df = add_readable_labels(predictions_df)

    #Build route adjacency
    adjacency = build_adjacency_list(metro_edges_df)

    #Generate possible routes
    possible_routes = generate_routes(adjacency,origin_station,destination_station,max_transfer,max_routes=5)

    if not possible_routes:
        return {
            "origin_station": origin_station,
            "destination_station": destination_station,
            "routes": [],
            "message": "No candidate routes found."
        }

    #Score routes
    rated_routes = score_routes(possible_routes, predictions_df)

    return {
        "origin_station": origin_station,
        "destination_station": destination_station,
        "routes": rated_routes
    }

#formatter
def format_route_suggestions(result):
    if not result["routes"]:
        return result["message"]

    lines = []
    lines.append(f"Origin: {result['origin_station']}")
    lines.append(f"Destination: {result['destination_station']}")
    lines.append("")

    for i, route_info in enumerate(result["routes"][:3], start=1):
        route_name = f"Route {chr(64 + i)}"

        avg_congestion = route_info["total_congestion_penalty"] / max(route_info["num_stops"], 1)
        avg_feeder = route_info["total_feeder_bonus"] / max(route_info["num_stops"], 1)

        lines.append(f"{route_name}")
        lines.append(f"  Stations: {route_info['route']}")
        lines.append(f"  Stops: {route_info['num_stops']}")
        lines.append(f"  Route score: {route_info['final_score']:.2f}")
        lines.append(f"  Avg congestion score: {avg_congestion:.2f}")
        lines.append(f"  Avg feeder score: {avg_feeder:.2f}")

        if route_info["station_details"]:
            end_station = route_info["station_details"][-1]
            lines.append(
                f"  Destination station congestion: {end_station['congestion_label']}"
            )
            lines.append(
                f"  Destination station feeder availability: {end_station['feeder_label']}"
            )

        lines.append("")

    return "\n".join(lines)
