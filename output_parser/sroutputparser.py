from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

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


template1 = PromptTemplate(
    template="Explain the {topic} in brief",
    input_variables=["topic"],
)

template2 = PromptTemplate(
    template="summarize the following text in 5 lines. \n {text}",
    input_variables=["text"],
)

# without using parser
# prompt1 = template1.invoke({"topic": "black hole"})

# result = model.invoke(prompt1)

# prompt2 = template2.invoke({"text": result.content})

# result = model.invoke(prompt2)
# print(result.content)


# using string parser

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser  # chain in langchain

result = chain.invoke({"topic": "black hole"})

print(result)
