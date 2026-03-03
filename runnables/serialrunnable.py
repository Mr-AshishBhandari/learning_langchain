from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.schema.runnable import RunnableSequence


from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    huggingfacehub_api_token=api_key,
)
model = ChatHuggingFace(llm=llm)


prompt = PromptTemplate(
    template="Generate a joke about {topic}", input_variables=["topic"]
)

parser = StrOutputParser()

runnable = RunnableSequence(prompt, model, parser)

result = runnable.invoke({"topic": "AI"})

print(result)


# Prompt --> Llm model --> Parser
#         (genetate a joke)
