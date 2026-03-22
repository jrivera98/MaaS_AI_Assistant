# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
#

# %% [markdown]
# # Creating GAT Journey Planner Agent Model
#
# ### Read in training data -> NetworkX graph

# %%
import pandas as pd
import networkx as nx

data_path = "C:\\Users\\sasab\\Documents\\Projects\\MaaS_AI\\Agents\\journey_planner\\data\\"
db_path = "C:\\Users\\sasab\\Documents\\Projects\\MaaS_AI\\Main_App\\data_bases\\"

#loading in csv file
route_edges_df = pd.read_csv(data_path+"pune_maas_journey_planner_data.csv") #pune_maas_journey_planner_data.csv - metro routing data table
timetable_l1 = pd.read_csv(db_path+"purple_line_timetable.csv") #timetable_data.csv - metro train timetable data
timetable_l2 = pd.read_csv(db_path+"aqua_line_timetable.csv")
timetable_l3 = pd.read_csv(db_path+"pink_line_timetable.csv")
swipes_df = pd.read_csv(db_path+"swipe_metadata.csv")  #swipe_metadata.csv - metro swipe data with each ticket pur

# clean columns data
route_edges_df.columns = route_edges_df.columns.str.strip()
timetable_l1.columns = timetable_l1.columns.str.strip()
timetable_l2.columns = timetable_l2.columns.str.strip()
timetable_l3.columns = timetable_l3.columns.str.strip()
swipes_df.columns = swipes_df.columns.str.strip()

# fix differnt name - normalize station
def clean_station_name(s):
    if pd.isna(s):
        return s
    return str(s).strip()

# Clean edge df station names
route_edges_df["station_name_from"] = route_edges_df["station_name_from"].apply(clean_station_name)
route_edges_df["station_name_to"] = route_edges_df["station_name_to"].apply(clean_station_name)

# Clean swipe df station names
swipes_df["Start station"] = swipes_df["Start station"].apply(clean_station_name)
swipes_df["End Station"] = swipes_df["End Station"].apply(clean_station_name)
swipes_df["Swipe In station"] = swipes_df["Swipe In station"].apply(clean_station_name)

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

for col in ["Start station", "End Station", "Swipe In station"]:
    swipes_df[col] = swipes_df[col].apply(apply_station_map)
    
# Parse swipe times 
swipes_df["Swipe in time"] = pd.to_datetime(
    swipes_df["Swipe in time"], format="%H:%M", errors="coerce"
)

swipes_df["Approximate boarding time"] = pd.to_datetime(
    swipes_df["Approximate boarding time"], format="%H:%M", errors="coerce"
)

swipes_df["swipe_hour"] = swipes_df["Swipe in time"].dt.hour
swipes_df["boarding_hour"] = swipes_df["Approximate boarding time"].dt.hour

#encode 'string' value attributes (line,mode)
#for every new value map to unique 
line_map = {name: i for i, name in enumerate(route_edges_df["line"].unique())}     #line: purple | pink | aqua
mode_map = {name: i for i, name in enumerate(route_edges_df["mode"].unique(), start=1)} # mode: metro | feeder bus

route_edges_df["line_id"] = route_edges_df["line"].map(line_map)
route_edges_df["mode_id"] = route_edges_df["mode"].map(mode_map)
route_edges_df["is_transfer"] = route_edges_df["is_transfer"].astype(int)
route_edges_df["bidirectional"] = route_edges_df["bidirectional"].astype(int)

#combine all timetable df
def reshape_timetable(tt_df, line_name):
    tt_long = tt_df.melt(
        id_vars=["Train ID", "Direction"],
        var_name="station_name",
        value_name="scheduled_time"
    )
    tt_long["station_name"] = tt_long["station_name"].apply(clean_station_name)
    tt_long["scheduled_time"] = pd.to_datetime(
        tt_long["scheduled_time"], format="%H:%M", errors="coerce"
    )
    tt_long["hour"] = tt_long["scheduled_time"].dt.hour
    tt_long["line_name"] = line_name
    return tt_long

purple_long = reshape_timetable(timetable_l1, "purple")
aqua_long = reshape_timetable(timetable_l2, "aqua")
pink_long = reshape_timetable(timetable_l3, "pink")

timetable_df = pd.concat([purple_long, aqua_long, pink_long], ignore_index=True)

#check
# print ("========route_edges table========")
# print (route_edges_df)

# print ("\n========timetable table========")
# print (timetable_df)

# print ("\n========swipes metadata table========")
# print (swipes_df)

# %% [markdown]
# ### Create a station master list
# | Feature            | Meaning      |
# | ------------------ | ------------ |
# | `transfer_station` | node feature |
# | `is_transfer`      | edge feature |
#

# %%
stations_from = route_edges_df[["station_id_from", "station_name_from","is_transfer"]].rename(
    columns={"station_id_from": "station_id", "station_name_from": "station_name"}
)

stations_to = route_edges_df[["station_id_to", "station_name_to"]].rename(
    columns={"station_id_to": "station_id", "station_name_to": "station_name"}
)

stations_df = pd.concat([stations_from, stations_to], ignore_index=True).drop_duplicates()
stations_df = stations_df.sort_values("station_id").reset_index(drop=True)

#add transfer flag
transfer_flags = (
    route_edges_df.groupby("station_id_from")["is_transfer"]
    .max()
    .reset_index()
    .rename(columns={"station_id_from": "station_id", "is_transfer": "transfer_station"})
)

stations_df = stations_df.merge(transfer_flags, on="station_id", how="left")
stations_df["transfer_station"] = stations_df["transfer_station"].fillna(0).astype(int)

# print(stations_df)

# %% [markdown]
# ### Building Feature
#

# %%
#count train stops at station per hr
trains_stopping = (
    timetable_df.groupby(["station_name", "hour"])
    .size()
    .reset_index(name="trains_stopping")
)

#define  swipe based features
# Boardings = people swiping into a station during an hour
boardings = (
    swipes_df.groupby(["Swipe In station", "swipe_hour"])
    .size()
    .reset_index(name="expected_boardings")
    .rename(columns={"Swipe In station": "station_name", "swipe_hour": "hour"})
)

# Alightings = people whose end station is that station during an hour
alightings = (
    swipes_df.groupby(["End Station", "boarding_hour"])
    .size()
    .reset_index(name="expected_alightings")
    .rename(columns={"End Station": "station_name", "boarding_hour": "hour"})
)

# Feeder requests = people requesting feeder service for that end station during an hour
feeder_requests = (
    swipes_df[
        swipes_df["Feeder Service (Y/N)"]
        .astype(str)
        .str.strip()
        .str.upper()
        .isin(["Y", "YES", "TRUE", "1"])
    ]
    .groupby(["End Station", "boarding_hour"])
    .size()
    .reset_index(name="feeder_requests")
    .rename(columns={"End Station": "station_name", "boarding_hour": "hour"})
)

#check
# print("=============train stop=============")
# print (trains_stopping)

# print("\n=============boardings=============")
# print (boardings)

# print("\n=============alightings=============")
# print (alightings)

# print("\n=============feeder requests=============")
# print (feeder_requests)


# %% [markdown]
# ### CREATE STATION-HOUR GRID

# %%
#every station should have a row for every hour in the timetable range
hours_df = pd.DataFrame({
    "hour": sorted(timetable_df["hour"].dropna().unique())
})

station_hour_df = (
    stations_df.assign(key=1)
    .merge(hours_df.assign(key=1), on="key")
    .drop("key", axis=1)
)
# print("\n=============Hours=============")
# print (hours_df)

# print("\n=============station_hour=============")
# print (station_hour_df)

# %%
#every station should have a row for every hour in the timetable range
hours_df = pd.DataFrame({
    "hour": sorted(timetable_df["hour"].dropna().unique())
})

station_hour_df = (
    stations_df.assign(key=1)
    .merge(hours_df.assign(key=1), on="key")
    .drop("key", axis=1)
)

# %% [markdown]
# ### Build Feat Table

# %%
#MERGE FEATURES INTO ONE TABLE
station_features_df = station_hour_df.merge(
    trains_stopping, on=["station_name", "hour"], how="left"
)

station_features_df = station_features_df.merge(
    boardings, on=["station_name", "hour"], how="left"
)

station_features_df = station_features_df.merge(
    alightings, on=["station_name", "hour"], how="left"
)

station_features_df = station_features_df.merge(
    feeder_requests, on=["station_name", "hour"], how="left"
)

for col in ["trains_stopping", "expected_boardings", "expected_alightings", "feeder_requests"]:
    station_features_df[col] = station_features_df[col].fillna(0).astype(int)

# %% [markdown]
# ### CREATE SIMPLE LABELS

# %%
# Simple congestion score for V1
station_features_df["congestion_score"] = (
    station_features_df["expected_boardings"]
    + station_features_df["expected_alightings"]
    + station_features_df["feeder_requests"]
    + station_features_df["transfer_station"]
)

def label_congestion(score):
    if score < 3:
        return 0   # low
    elif score < 8:
        return 1   # medium
    return 2       # high

station_features_df["congestion_label"] = station_features_df["congestion_score"].apply(label_congestion)


# Feeder label for V1
def label_feeder(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    elif n == 2:
        return 2
    return 3   # 3+

station_features_df["feeder_label"] = station_features_df["feeder_requests"].apply(label_feeder)



# %% [markdown]
# ###  SAVE OUTPUT
# can you as ref to see what data is being used to train the node features

# %%
# Keep only the simpler V1 columns
final_cols = [
    "station_id",
    "station_name",
    "hour",
    "transfer_station",
    "trains_stopping",
    "expected_boardings",
    "expected_alightings",
    "feeder_requests",
    "congestion_label",
    "feeder_label",
]

station_features_df = station_features_df[final_cols]

station_features_df.to_csv("station_features.csv", index=False)

# print("Saved station_features.csv")
# # print(station_features_df.head(20))
# print(station_features_df.shape)

# print(station_features_df["congestion_label"].value_counts())
# print(station_features_df["feeder_label"].value_counts())

# %% [markdown]
# ### Define NetworkX graph node/edge structure


# %% [markdown]
# ### Convert Network X graph to PyG dataset to train model

# %%

import torch
from torch_geometric.data import Data
import pandas as pd
# from torch_geometric.utils.convert import from_networkx
import torch.nn.functional as F
import torch_geometric.transforms as T


def create_pyg_graph(route_edges_df: pd.DataFrame,
                     station_features_df: pd.DataFrame):

    # Convert metro edges + station feature DataFrames into a PyTorch Geometric Data object

    # station index map
    station_ids = station_features_df["station_id"].tolist()

    station_to_idx = {
        station_id: idx
        for idx, station_id in enumerate(station_ids)
    }

    # node feature matrix
    feature_cols = [
        col for col in station_features_df.columns
        if col != "station_id"
        and pd.api.types.is_numeric_dtype(station_features_df[col])
    ]

    x = torch.tensor(
        station_features_df[feature_cols].values,
        dtype=torch.float
    )

    # edge index
    valid_edges = route_edges_df[
        route_edges_df["station_id_from"].isin(station_to_idx) &
        route_edges_df["station_id_to"].isin(station_to_idx)
    ]

    source_nodes = valid_edges["station_id_from"].map(station_to_idx)
    target_nodes = valid_edges["station_id_to"].map(station_to_idx)

    edge_pairs = list(zip(source_nodes, target_nodes))

    # Add reverse edges
    edge_pairs += [(dst, src) for src, dst in edge_pairs]

    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()

    # adding edge attributes
    edge_feature_cols = [
        col for col in route_edges_df.columns
        if col not in ["station_id_from", "station_id_to"]
        and pd.api.types.is_numeric_dtype(route_edges_df[col])
    ]

    edge_attr = None

    if edge_feature_cols:

        edge_values = valid_edges[edge_feature_cols].values
        edge_values = pd.concat(
            [pd.DataFrame(edge_values), pd.DataFrame(edge_values)],
            ignore_index=True
        ).values

        edge_attr = torch.tensor(edge_values, dtype=torch.float)

    # Create PyG Data object
    data = Data(
        x=x,
        edge_index=edge_index
    )

    if edge_attr is not None:
        data.edge_attr = edge_attr

    data.station_ids = station_ids
    data.node_feature_names = feature_cols
    data.edge_feature_names = edge_feature_cols
    data.y_congestion = torch.tensor(
                                station_features_df["congestion_label"].values,
                                dtype=torch.long
                                )
    data.y_feeder = torch.tensor(
                                station_features_df["feeder_label"].values,
                                dtype=torch.long
                                )

    return data


pyg_data = create_pyg_graph(route_edges_df, station_features_df)

# print(pyg_data)
# print("Edge index shape:", pyg_data.edge_index.shape)
# print("Edge attr shape:", pyg_data.edge_attr.shape)
# print("num of node features:", pyg_data.num_node_features)
# print("y_congestion :", pyg_data.y_congestion)
# print("num of node features:", pyg_data.y_feeder)



# %% [markdown]
# ### Create GAT model
#

# %%
from torch_geometric.nn import GATv2Conv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class MultiTaskGAT(nn.Module):
    # in_channels - number of node feature
    # hidden_channels - 16
    # out_channels - number of classes output channels.set to 3, will rep congestion level to use in Dispact Agent layer
    # edge_features - number of edge feature
    def __init__(self, in_channels, hidden_channels, edge_features,congestion_classes=3, feeder_classes=4):
        super().__init__()
        # First GAT layer with multi-head attention
        # self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=8, edge_dim=edge_features)
        # Second GAT layer (output layer)
        # self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6)
        self.conv2 = GATv2Conv(hidden_channels * 8,hidden_channels,heads=1,concat=False,edge_dim=edge_features)

        self.congestion_head = nn.Linear(hidden_channels, congestion_classes)
        self.feeder_head = nn.Linear(hidden_channels, feeder_classes)

    #dropout - default .6 or 60% once outr dataset is small and doent have alot of node feature
    #new default set to 0.1-0.2 for minimal regularization and small graphs
    def forward(self, x, edge_index, edge_attr):
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.conv2(x, edge_index, edge_attr))

        congestion_logits = self.congestion_head(x)
        feeder_logits = self.feeder_head(x)

        return congestion_logits, feeder_logits

#create GAT model
num_classes =  3
multi_gat_model = MultiTaskGAT(pyg_data.num_node_features, 16,pyg_data.edge_attr.shape[1])
optimizer = torch.optim.Adam(multi_gat_model.parameters(), lr=0.005, weight_decay=5e-4)

# Sanity checks
def test_check():
    print("=== Input Graph Shapes ===") #orginal shape
    print("x shape(num of node):", pyg_data.x.shape) 
    print("num of node:", pyg_data.num_nodes)
    print("edge_index shape:", pyg_data.edge_index.shape)
    print("edge_attr shape:", pyg_data.edge_attr.shape)
    print("edge_attr shape:", pyg_data.edge_attr.shape)
    
    multi_gat_model.eval()
    with torch.no_grad():
        # out = multi_gat_model(pyg_data,pyg_data.edge_attr.shape) #gat model shape
        congestion_logits, feeder_logits = multi_gat_model(pyg_data.x,pyg_data.edge_index,pyg_data.edge_attr) #gat model shape without gradient change
        print("\n=== Model Output Shape ===")
        print("congestion_logits shape:", congestion_logits.shape)
        print("feeder_logits shape:", feeder_logits.shape)
    
# test_check()



# %% [markdown]
# ### creating Trainning/Val/Test masks
#
# Total nodes = 1470
#
# | Split      | Percentage | Nodes        |
# | ---------- | ---------- | ------------ |
# | Training   | **70%**    | **1029 nodes** |
# | Validation | **15%**    | **221 nodes** |
# | Test       | **15%**    | **220 nodes** |
#

# %%
num_nodes = pyg_data.num_nodes

train_size = int(num_nodes*0.7)
val_size = int(num_nodes*0.15)
max_size = train_size+val_size

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
val_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)

rand_perm = torch.randperm(num_nodes)

train_mask[rand_perm[:train_size]] = True
val_mask[rand_perm[train_size:max_size]] = True
test_mask[rand_perm[max_size:]] = True

pyg_data.train_mask = train_mask
pyg_data.val_mask = val_mask
pyg_data.test_mask = test_mask

print(f"num_train_nodes: {sum(train_mask)} | num_val_nodes: {sum(val_mask)} | num_test_nodes: {sum(test_mask)}")


# %%
# print("=== Orginal datatypes ===")
# print("x dtype:", pyg_data.x.dtype)
# print("edge_index dtype:", pyg_data.edge_index.dtype)
# print("edge_attr dtype:", pyg_data.edge_attr.dtype)

#convert to correct datatypes
if pyg_data.x.dtype != torch.float32: 
    pyg_data.x = pyg_data.x.float()
    print("x dtype:", pyg_data.x.dtype)
    
if pyg_data.edge_index.dtype != torch.int64:
    pyg_data.edge_index = pyg_data.edge_index.float()
    print("edge_index dtype:", pyg_data.edge_index.dtype)
    
if pyg_data.edge_attr.dtype != torch.float32: 
    pyg_data.edge_attr = pyg_data.edge_attr.long()
    print("edge_attr dtype:", pyg_data.edge_attr.dtype)
    

print("\n=== Corrected datatypes ===")


# %% [markdown]
# ### Create training/test eval functions

# %%
def train():
    multi_gat_model.train()
    optimizer.zero_grad()
    # Forward pass
    out1, out2 = multi_gat_model(pyg_data.x,pyg_data.edge_index,pyg_data.edge_attr)
    
    # Calculate loss on training nodes only
    loss1 = F.cross_entropy(out1[train_mask], pyg_data.y_congestion[train_mask])
    loss2 = F.cross_entropy(out2[train_mask], pyg_data.y_feeder[train_mask])
    
    tot_loss = loss1+ loss2
    tot_loss.backward()
    optimizer.step()
    return tot_loss.item()

#test function to test the trainning accuracy
def test(model):
    model.eval()
    with torch.no_grad():      
        #preiction on the whole graph
        congestion_logits, feeder_logits = model(pyg_data.x,pyg_data.edge_index,pyg_data.edge_attr)

        #only focus on training_mask
        loss1 = F.cross_entropy(congestion_logits[test_mask], pyg_data.y_congestion[test_mask])
        loss2 = F.cross_entropy(feeder_logits[test_mask], pyg_data.y_feeder[test_mask])
        
        # check

        # print("out1 shape:", congestion_logits.shape)
        # print("out2 shape:", feeder_logits.shape)
        # print("y_congestion shape:", pyg_data.y_congestion.shape)
        # print("y_feeder shape:", pyg_data.y_feeder.shape)

        # print("out1 train shape:", congestion_logits[test_mask].shape)
        # print("out2 train shape:", feeder_logits[test_mask].shape)
        # print("y_congestion train shape:", pyg_data.y_congestion[test_mask].shape)
        # print("y_feeder train shape:", pyg_data.y_feeder[test_mask].shape)
        # #
        
        tot_loss = loss1+ loss2
        
        #compute accuracy
        pred1 = congestion_logits.argmax(dim=-1)
        correct1 = (pred1[test_mask] == pyg_data.y_congestion[test_mask]).sum()
        acc1 = int(correct1) / int(test_mask.sum())
        
        pred2 = feeder_logits.argmax(dim=-1)
        correct2 = (pred2[test_mask] == pyg_data.y_feeder[test_mask]).sum()
        acc2 = int(correct2) / int(test_mask.sum())
        
        return acc1,acc2
    
# funct to vaildate model
def evaluate(model):
    model.eval()
    with torch.no_grad():      
        #preiction on the whole graph
        congestion_logits, feeder_logits = model(pyg_data.x,pyg_data.edge_index,pyg_data.edge_attr)

        #only focus on training_mask
        loss1 = F.cross_entropy(congestion_logits[val_mask], pyg_data.y_congestion[val_mask])
        loss2 = F.cross_entropy(feeder_logits[val_mask], pyg_data.y_feeder[val_mask])
        
        tot_loss = loss1+ loss2
        
        #compute accuracy
        pred1 = congestion_logits.argmax(dim=-1)
        correct1 = (pred1[val_mask] == pyg_data.y_congestion[val_mask]).sum()
        acc1 = int(correct1) / int(val_mask.sum())

        #compute accuracy
        pred2 = feeder_logits.argmax(dim=-1)
        correct2 = (pred2[val_mask] == pyg_data.y_feeder[val_mask]).sum()
        acc2 = int(correct2) / int(val_mask.sum())
                
        # pred2 = feeder_logits
        return acc1,acc2, tot_loss

epoch_num = 95

# Training loop over epochs
for epoch in range(epoch_num):
    train_loss = train()
    train_acc1, train_acc2 = test(multi_gat_model)
    val_acc1,val_acc2, val_loss = evaluate(multi_gat_model)
    
    if epoch%5==0 or epoch == epoch_num-1:
        print(f"\nEpoch: {epoch} | train_loss: {train_loss:.4f} | train_congestion_acc: {train_acc1*100:.2f}% | train_Feeder_acc: {train_acc2*100:.2f}% |" f" val_loss: {val_loss:.4f} | val_congestion_acc: {val_acc1*100:.2f}| val_Feeder_acc: {val_acc2*100:.2f}%")
        


# %% [markdown]
# # Feature
# | Name               | Feature 1  |    Feature 2     | Feature 3   | Feature 4  |     Feature 5  |
# | ------------------ | ---------- | ---------------- | ---------   | ---------- | ---------------|
# | Node               | station_id | transfer_station | boardings   | alightings | feeder_requests|
# | Edge               | from       | to               | travel_time | line       | is_transfer    |
# | Label              | station    | congestion_class | feeder_class                              |
#
