from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")


llm = HuggingFaceEndpoint(
    repo_id="zai-org/GLM-5",
    task="text-generation",
    huggingfacehub_api_token=api_key,
)

model = ChatHuggingFace(llm=llm)
chat_history = [
    SystemMessage("You are a helpful ai assistant"),
]

while True:
    user_input = input("You:")
    if user_input == "exit":
        break
    chat_history.append(HumanMessage(user_input))
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(result.content))
    print("AI:", result.content)
