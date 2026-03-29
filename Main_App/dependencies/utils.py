# import torch
# import pandas as pd
# import json
# from torch_geometric.data import Data
# import torch.nn.functional as F
# import networkx as nx
# from pandasql import sqldf
# from collections import deque,defaultdict
# import re

# from models.GAT.gat_journey_planner import create_pyg_graph
# from pipeline.inference import load_station_master,run_multitask_inference,add_readable_labels,add_swipe_info,load_pyg_data,load_trained_model,load_data
# from pipeline.pipelines import get_Station_names, build_pyg_graph,set_trip_schema
# from pipeline.routing import congestion_penalty, feeder_bonus,recommend_routes
# from pipeline.formatter import format_route_suggestions
# from pipeline.time import get_time_table, get_boarding_time,get_arrival_time

#
TRIP_SCHEMA = {
        "start_station": None,
        "end_station": None,
        "start_station_id": None,
        "end_station_id": None,
        "feeder_required": None,   # true / false / null
        "feeder_type": None,       # bike / bus / shuttle / null
        "departure_time": None,
        "arrival_time": None
        # "start_location": None,
        # "final_destination": None
}

REQUIRED_FIELDS = ["start_station", "end_station", "feeder_required"]

#global var
PRO_DIR = "C:\\Users\\sasab\\Documents\\Projects\\MaaS_AI\\Main_App\\"
DATA_PATH = "models\\GAT\\data\\"


