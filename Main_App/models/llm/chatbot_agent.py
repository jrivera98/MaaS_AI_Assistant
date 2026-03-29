import requests 
import json
from datetime import datetime

EXTRACTION_SYS_MSG ="""
            You extract trip-planning information from user messages for a Mobility-as-a-Service assistant.

            Return ONLY valid JSON with these keys:
            - start_station
            - end_station
            - feeder_required
            - feeder_type
            - departure_time
            - arrival_time
            output: {"start_station": null, "end_station": null,"feeder_required": null,"feeder_type": null,"departure_time": null,"arrival_time": null}


            Rules:
            - Use null if the user did not provide a value.
            - feeder_required must be true, false, or null.
            - feeder_type must be one of: "bike", "bus", "shuttle", or null.
            - Do not invent values.
            - If the user says they do not need feeder service, set feeder_required to false and feeder_type to null.
            - If the user says they need feeder service but gives no type, set feeder_required to true and feeder_type to null.
            - Return JSON only. No markdown, no explanation.
"""
        
FROMATTING_SYS_MSG ="""
                    You are an intelligent assistant for a Mobility-as-a-Service system.
                    Help users with routing, congestion insights, determine feed demand, transit coordination,
                    station demand explanations, and operational recommendations.
                    Request that the user supples the following required infomation: start station, end station, if they require Feeder vehicle services, and if yes which type of vehicle.
                    and optional infomation: departure time, arrival time, start location, final destination. if user only supples part of the information ask follows questions untill required all information is recieved.
"""

class LocalChatbot:
    def __init__(self, model="qwen2.5:7b"):
        self.model = model
        self.ollama_url= "http://localhost:11434/api/generate"
        self.conversation_history =[]
        
    def chat(self, user_message):
        """Single turn conversation"""
        self.conversation_history.append({
        "role": "user",
        "content": user_message
        })

        context = (
                    "You are an intelligent assistant for a Mobility-as-a-Service system."
                    "Help users with routing, congestion insights, determine feed demand, transit coordination,"
                    "station demand explanations, and operational recommendations."
                    "Request that the user supples the following required infomation: start station, end station, if they require Feeder vehicle services, and if yes which type of vehicle."
                    "and optional infomation: departure time, arrival time, start location, final destination. if user only supples part of the information ask follows questions untill required all information is recieved."+"\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in self.conversation_history[-7:] # Last 7 messages
        ]))
        
        try:
            # Call local model
            response = requests.post(
                self.ollama_url,
                json={
                "model": self.model,
                "prompt": context,
                "stream": False,
                "temperature": 0.7,
                },
                timeout=60
            )
        
            response.raise_for_status()
            answer = response.json().get("response", "No response")
        except requests.exceptions.RequestException as e:
            answer = f"Error calling Ollama: {e}"
            
        self.conversation_history.append({
        "role": "assistant",
        "content": answer,
        "system message":"You are an intelligent assistant for a Mobility-as-a-Service system. "
                            + "Help users with routing, congestion insights, determine feed demand, transit coordination, "
                            +"station demand explanations, and operational recommendations."
        })

        return answer


class Chatbot:
    def __init__(self,system_message, model="qwen2.5:7b"):
        self.model = model
        self.ollama_url= "http://localhost:11434/api/generate"
        self.conversation_history =[]
        self.sys_message = system_message
        
    def chat(self, user_message):
        """Single turn conversation"""
        self.conversation_history.append({
        "role": "user",
        "content": user_message
        })
        
        context = (self.sys_message+"\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in self.conversation_history[-7:] # Last 7 messages
        ]))
        
        try:
            # Call local model
            response = requests.post(
                self.ollama_url,
                json={
                "model": self.model,
                "prompt": context,
                "stream": False,
                "temperature": 0.7,
                },
                timeout=60
            )
        
            response.raise_for_status()
            answer = response.json().get("response", "No response")
        except requests.exceptions.RequestException as e:
            answer = f"Error calling Ollama: {e}"
            
        self.conversation_history.append({
        "role": "assistant",
        "content": answer
        })

        return answer

       
# def main():
#     print("starting chatbot script...\n")
    
#     bot = LocalChatbot(model="qwen2.5:7b")
#     while True:
#         user_input = input("You: ")
        
#         if user_input.lower() in ["exit", "quit"]:
#             break
            
#         response = bot.chat(user_input)
#         print(f"Bot: {response}\n")

# #Usage
# if __name__ == "__main__":
#     main()

# You extract trip-planning information from user messages for a Mobility-as-a-Service assistant.

# Return ONLY valid JSON with these keys:
# - start_station
# - end_station
# - feeder_required
# - feeder_type
# - departure_time
# - arrival_time
# - start_location
# - final_destination

# Rules:
# - Use null if the user did not provide a value.
# - feeder_required must be true, false, or null.
# - feeder_type must be one of: "bike", "bus", "shuttle", or null.
# - Do not invent values.
# - If the user says they do not need feeder service, set feeder_required to false and feeder_type to null.
# - If the user says they need feeder service but gives no type, set feeder_required to true and feeder_type to null.
# - Return JSON only. No markdown, no explanation.
# - Request that the user supples the following required infomation: start_station, end_station, feeder_required, feeder_type(only if feeder_required is true)"
# - optional infomation: departure_time, arrival_time, start_location, final_destination.
# - if user only supples part of the information ask follows questions untill all information is recieved
