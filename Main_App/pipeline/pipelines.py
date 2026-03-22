import torch
import pandas as pd
import json
from torch_geometric.data import Data
import torch.nn.functional as F
from collections import deque,defaultdict
import networkx as nx
from models.GAT.gat_journey_planner import create_pyg_graph
from pandasql import sqldf

#global var
pro_dir = "C:\\Users\\sasab\\Documents\\Projects\\MaaS_AI\\Main_App\\"
data_path = "models\\GAT\\data\\"

#graph_builder
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
def clean_station_name(s):
    if pd.isna(s):
        return s
    return str(s).strip()

#load station master list

#load in csv and cleans col data
def load_data(csv_path, col_to_clean):
    data_path = csv_path
        
    #loading in csv file
    df = pd.read_csv(data_path) 

    #clean columns data
    df.columns = df.columns.str.strip()
    
    for col in col_to_clean:
         # Clean edge df station names
        df[col] = df[col].apply(clean_station_name)
    
    return df

def load_station_master():
    col = ["station_name"]
    master_csv_path = pro_dir+"data_bases\\master_stations_list.csv"
    master_df = load_data(master_csv_path, col)
    
    return master_df


def  load_pyg_data():
    metro_edges_path = pro_dir+data_path+"pune_maas_journey_planner_data.csv"
    station_features_path = pro_dir+data_path+"station_features.csv"
    
    
    #loading in csv file
    route_edges_df = pd.read_csv(metro_edges_path) #pune_maas_journey_planner_data.csv - metro routing data table
    station_feat_df = pd.read_csv(station_features_path) #timetable_data.csv - metro train timetable data

    # clean columns data
    route_edges_df.columns = route_edges_df.columns.str.strip()
    station_feat_df.columns = station_feat_df.columns.str.strip()

    # fix differnt name - normalize station
    def clean_station_name(s):
        if pd.isna(s):
            return s
        return str(s).strip()

    # Clean edge df station names
    route_edges_df["station_name_from"] = route_edges_df["station_name_from"].apply(clean_station_name)
    route_edges_df["station_name_to"] = route_edges_df["station_name_to"].apply(clean_station_name)
    station_feat_df["station_name"] = station_feat_df["station_name"].apply(clean_station_name)
    
    # Map alternate names to one standard name
    station_name_map = {
        "RamWadi": "Ramwadi",
        "Ruby Hall": "Ruby Hall Clinic",
        "Civil Court": "District Court (Civil Court)",
        "District Court Pune": "District Court (Civil Court)",
    }

    def apply_station_map(s):
        if pd.isna(s):
            return s
        return station_name_map.get(s, s)


    #encode string attributes
    #for every new value map to unique 
    line_map = {name: i for i, name in enumerate(route_edges_df["line"].unique())}     #line: purple | pink | aqua
    mode_map = {name: i for i, name in enumerate(route_edges_df["mode"].unique(), start=1)} # mode: metro | feeder bus

    route_edges_df["line_id"] = route_edges_df["line"].map(line_map)
    route_edges_df["mode_id"] = route_edges_df["mode"].map(mode_map)
    route_edges_df["is_transfer"] = route_edges_df["is_transfer"].astype(int)
    route_edges_df["bidirectional"] = route_edges_df["bidirectional"].astype(int)
    
    #check
    # print ("========route_edges table========")
    # print (route_edges_df)

    # print ("\n========timetable table========")
    # print (timetable_df)

    # print ("\n========swipes metadata table========")
    # print (swipes_df)
    
    #convert df to pyg df
    pyg_data = create_pyg_graph(route_edges_df,station_feat_df)   
        
    return route_edges_df, station_feat_df, pyg_data

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
    label_df["congestion_label"] = dataframe["pred_congestion_class"].map(congestion_penalty)
    label_df["feeder_label"] = dataframe["pred_feeder_class"].map(feeder_bonus)
    return label_df

#routing 
def build_adjacency_list(metro_edges_df):
    adjacency_station = defaultdict(list)

    for _, row in metro_edges_df.iterrows():
        a = row["station_id_from"]
        b = row["station_id_to"]

        adjacency_station[a].append(b)
        adjacency_station[b].append(a)

    # print("===========================\n")
    # print("adjacency_graph:\n")
    # print(adjacency_station)
    
    return adjacency_station

def generate_routes(adjacency, origin_station, destination_station, max_routes=5):
    routes = []
    queue = deque([[origin_station]])

    while queue and len(routes) < max_routes:
        path = queue.popleft()
        current = path[-1]

        if current == destination_station:
            routes.append(path)
            continue

        # if len(path) > max_transfer:
        #     continue

        for neighbor in adjacency[current]:
            if neighbor not in path:
                new_path = path + [neighbor]
                queue.append(new_path)
    return routes

def congestion_penalty(congestion_class):
    mapping = {
        0: 0.0,   # low
        1: 3.0,   # medium
        2: 8.0    # high
    }
    return mapping.get(congestion_class, 0.0)

def feeder_bonus(feeder_class):
    mapping = {
        0: 0.0,
        1: 1.0,
        2: 2.0,
        3: 3.0
    }
    return mapping.get(feeder_class, 0.0)

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

        station_names_route = get_Station_names(route)
        print("-------------station_names_route---------------")
        print(station_names_route)
        
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

def recommend_routes(origin_station,destination_station,metro_edges_df,station_features_df,model):# max_transfer):
    # Build graph
    pyg_data = build_pyg_graph(metro_edges_df, station_features_df)

    #Run GAT predictions
    predictions_df, _, _ = run_multitask_inference(model, pyg_data)
    
    # print("===========================\n")
    # print("predictions_df:\n")
    # print(predictions_df)
    predictions_df = add_readable_labels(predictions_df)

    #Build route adjacency
    adjacency = build_adjacency_list(metro_edges_df)

    #Generate possible routes
    possible_routes = generate_routes(adjacency,origin_station,destination_station,max_routes=5)

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

# data normalization
def normalize_trip_info(parsed_responce):
    json_res = json.loads(parsed_responce)
    
    normalized = {
        "start_station": json_res.get("start_station"),
        "end_station": json_res.get("end_station"),
        "feeder_required": json_res.get("feeder_required"),
        "feeder_type": json_res.get("feeder_type"),
        "departure_time": json_res.get("departure_time"),
        "arrival_time": json_res.get("arrival_time"),
        # "start_location": json_res.get("start_location"),
        # "final_destination": json_res.get("final_destination")
    }

    if normalized["feeder_type"] is not None:
        normalized["feeder_type"] = str(normalized["feeder_type"]).lower().strip()

    if normalized["feeder_type"] not in {None, "bike", "rickshaw"}:
        normalized["feeder_type"] = None

    if normalized["feeder_required"] not in {True, False, None}:
        normalized["feeder_required"] = None

    return normalized

def get_missing_fields(context, req_fields):
    missing = []
    is_missing = True

    for field in req_fields:
        if context.get(field) is None:
            missing.append(field)

    if context.get("feeder_required") is True and context.get("feeder_type") is None:
        missing.append("feeder_type")
        
    if not missing:
        # followup_en = False
        is_missing = False

    return missing, is_missing

def build_followup_question(missing_fields):
    prompts = []    

    if "start_station" in missing_fields:
        prompts.append("What station are you starting from?")

    if "end_station" in missing_fields:
        prompts.append("What station are you heading to?")

    if "feeder_required" in missing_fields:
        prompts.append("Do you need a feeder vehicle for this trip?")

    if "feeder_type" in missing_fields:
        prompts.append("What type of feeder vehicle would you like: bike or rickshaw?")
    
    return " ".join(prompts)

def get_Station_names(route):
    stations = []
    
    station_master_df = load_station_master()

    # for route_info in ranked_sugg_routes:
    #     route_path = route_info #route_info["route"]
    station_name = []
        
        # for station_id in route_path:
    for index in range(len(route)):      
        station_id = route[index]
        # print(station_id)
        
        try: 
            station_name.append(station_master_df.loc[
                station_master_df["master_station_id"] == station_id, "station_name"
                ].iloc[0])
            
        except IndexError:
            # No match found
            # return "Station id: {station_id} was not found"
            print(f"Station id: {station_id} was not found")
        # if index == len(route)-1:#last index
        #     # print(station_name)
        #     stations.append(station_name)
            
    # print(station_name)
    return station_name
