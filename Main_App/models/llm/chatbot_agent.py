import requests 
import json
from datetime import datetime

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

        # Build context from history
        context = (
                    "You are an intelligent assistant for a Mobility-as-a-Service system."
                   "Help users with routing, congestion insights, determine feed demand, transit coordination,"
                    "station demand explanations, and operational recommendations."+"\n".join([
        f"{msg['role']}: {msg['content']}"
        for msg in self.conversation_history[-5:] # Last 5 messages
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
        except requests.exceptions.requestException as e:
            answer = f"Error calling Ollama: {e}"
            
        self.conversation_history.append({
        "role": "assistant",
        "content": answer,
        "system message":"You are an intelligent assistant for a Mobility-as-a-Service system. "
                            + "Help users with routing, congestion insights, determine feed demand, transit coordination, "
                            +"station demand explanations, and operational recommendations."
        })

        return answer

    
def main():
    print("starting chatbot script...\n")
    
    bot = LocalChatbot(model="qwen2.5:7b")
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ["exit", "quit"]:
            break
            
        response = bot.chat(user_input)
        print(f"Bot: {response}\n")

#Usage
if __name__ == "__main__":
    main()