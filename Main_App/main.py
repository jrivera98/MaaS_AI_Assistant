from pipeline.pipelines import recommend_routes,load_pyg_data, load_trained_model, format_route_suggestions, normalize_trip_info, get_missing_fields,build_followup_question,set_trip_schema,add_swipe_info
from models.GAT.gat_journey_planner import MultiTaskGAT
from datetime import datetime   
from models.llm.chatbot_agent import Chatbot,EXTRACTION_SYS_MSG,FROMATTING_SYS_MSG
import re

# from dependencies.imports import PRO_DIR,DATA_PATH,TRIP_SCHEMA
# from dependencies.imports import re
# from dependencies.imports import set_trip_schema, add_swipe_info,load_pyg_data,load_trained_model,recommend_routes,format_route_suggestions

# from pipeline.inference import add_swipe_info,load_pyg_data,load_trained_model
# from pipeline.pipelines import set_trip_schema
# from pipeline.routing import recommend_routes
# from pipeline.formatter import format_route_suggestions
# from pipeline.time import get_time_table, get_boarding_time,get_arrival_time

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

user_info={
    "name":"Denise",
    "age": "27",
    "gender":"female"
}
def main():
    metro_edges_path = PRO_DIR+DATA_PATH+"pune_maas_journey_planner_data.csv"
    station_features_path = PRO_DIR+DATA_PATH+"station_features.csv"
    checkpoint_path = PRO_DIR+"models\\checkpoint\\gat_maas_model.pt"
    recieved_required = False
    normalize_res = ""
    
    #define llms
    extraction_bot = Chatbot(EXTRACTION_SYS_MSG,model="qwen2.5:7b")
    formatter_bot = Chatbot(FROMATTING_SYS_MSG,model="qwen2.5:7b")
                
    print("\n-----------------------------------------------------------------------------------\n")  
    print("--------------------------------------")  
    print("to end program enter 'exit' or 'quit'")
    print("--------------------------------------\n")  
    print("\nWelcome! I’ll help you find the best route based on congestion levels and feeder availability\n")

    #user input
    request_mess = """
    “Let’s plan your trip\n\n
     First, I’ll need a few details:
     
     What station are you starting from?\n
     What station are you heading to?\n
     Do you need a feeder service (bike or rickshaw)?\n\n
     
     You can also include:\n
     departure or arrival time\n
     your exact starting or final destination\n
    """
    print(request_mess)
    
    #user_input = "Need to get from Bhakti Shakti to Ruby Hall Clinic and i need a bike"
    user_input = "Need to get from Bhakti Shakti to Baner, i need a bike, and I want to leave at 6:10 am"
    
    # loop until user exist convo
    #while True:  
    # user_input = input("You: ")
                    
    #loop until all required info is recieved
    # finalized_res = set_trip_schema(extraction_bot,user_input)
    # print(finalized_res)
    while True: 
        
        if user_input.lower() in ["exit", "quit"]:
            break
                        
        bot_response = extraction_bot.chat(user_input)
        # print(f"Bot: {bot_response}\n")
        # print("RAW RESPONSE:", repr(response))
        
        parse_res = re.split(r'([{}])', bot_response)
        cleaned_res = "{"+parse_res[2]+"}"
        print(cleaned_res)
        
        normalize_res = normalize_trip_info(cleaned_res)
        missing_data, followup_required = get_missing_fields(normalize_res, REQUIRED_FIELDS)
                
        if not followup_required:
            print("All req info is obtained")
            break
        else:
            followup_questions = build_followup_question(missing_data)
            print("Bot:")
            print(followup_questions)
            user_input = input("You: ")
            
            # user_input = "follow up question:"+followup_questions + "and user response:"+ followup_rep
            # print(user_input)
        
        
    # if user_input.lower() in ["exit", "quit"]:
    #     break
    # ------------
    # test input
    
    #----
    start_station_name = normalize_res["start_station"] #"Bhakti Shakti"#input("Start station>")
    final_station_name = normalize_res["end_station"] #"Shivaji Nagar"#input("End station>")
    # departure_time = input("Departure time>")
    # # desired_arrival_time = input("Desired arrival time>")    
    # # start_des = input("What is your start destination>") 
    # # final_des = input("What is your final destination>") 
    # # feeder_en = input("Do you require a Feeder service (y/yes or n/no)>") 
    # # feeder_type = input("Feeder vehicle prefrence(bike/rickshaw)>") 
            
    time_constraints = {
        "start": datetime.strptime("06:00", "%H:%M"),
        "end": datetime.strptime("06:52", "%H:%M")
    }
    
    # # metro_edges_df, station_features_df, pyg_gat_data ,station_id_to,station_id_from= load_pyg_data()
    metro_edges_df, station_features_df, pyg_gat_data = load_pyg_data()
    
    station_id_from = metro_edges_df.loc[
        metro_edges_df["station_name_from"] == start_station_name, "station_id_from"
        ].iloc[0]
    
    station_id_to = metro_edges_df.loc[
        metro_edges_df["station_name_to"] == final_station_name, "station_id_to"
        ].iloc[0]
            
    print("-------station_from--------\n")
    print(start_station_name)
    print(station_id_from)
    
    
    print("-------station_to--------\n")
    print(final_station_name)
    print(station_id_to)
    
    print("-------TRIP SCHEMA--------\n")
    TRIP_SCHEMA["start_station_id"] = station_id_from
    TRIP_SCHEMA["end_station_id"] = station_id_to
    TRIP_SCHEMA["start_station"] = start_station_name
    TRIP_SCHEMA["end_station"] = final_station_name
    TRIP_SCHEMA["feeder_required"] = normalize_res["feeder_required"]
    TRIP_SCHEMA["feeder_type"] = normalize_res["feeder_type"]
    TRIP_SCHEMA["departure_time"] = normalize_res["departure_time"]
    TRIP_SCHEMA["arrival_time"] = normalize_res["arrival_time"]      
    print(TRIP_SCHEMA)
    
    #update swipe db with user and trip details 
    add_swipe_info(user_info,TRIP_SCHEMA)
    #then load model with updated info 
    #get best route
    #then use that and time. if arrival time then determine what depart time the user will need to leave.
    #if depart time then determine when arrive time
        #if feeder is required call service to be there by arrive time               
    model_kwargs ={
        "in_channels":pyg_gat_data.num_node_features,
        "hidden_channels":16, 
        "edge_features": pyg_gat_data.edge_attr.shape[1]
    }
    
    maas_gat_model, model_metadata = load_trained_model(MultiTaskGAT, checkpoint_path,model_kwargs)
    
    result = recommend_routes(
    origin_station=TRIP_SCHEMA["start_station_id"],
    destination_station=TRIP_SCHEMA["end_station_id"],
    metro_edges_df=metro_edges_df,
    station_features_df=station_features_df,
    model=maas_gat_model
    )

    #print(result)
    formatted_routes = format_route_suggestions(result,TRIP_SCHEMA)
    print(formatted_routes)
    
    # # # bot = LocalChatbot(model="qwen2.5:7b")
    # # # #  while True:
    # # # #     user_input = input("You: ")
        
    # # # #     if user_input.lower() in ["exit", "quit"]:
    # # # #         break
            
    # # # #     response = bot.chat(user_input)
    # # # #     print(f"Bot: {response}\n")
    
    user_input = "Take these route suggestions from the routing agent and write out a response for the user."+formatted_routes
    print(user_input)
    response = formatter_bot.chat(user_input)
    print(f"Bot: {response}\n")
    
    # user_input = input("You:")

if __name__ == "__main__":
    main()
