from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.schema.runnable import (
    RunnableSequence,
    RunnablePassthrough,
    RunnableParallel,
)

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

prompt1 = PromptTemplate(
    template=" Generate a joke on {topic}", input_variables=["topic"]
)

prompt2 = PromptTemplate(template="explain the joke {topic}", input_variables=["topic"])

parser = StrOutputParser()

sequential_runnable = RunnableSequence(prompt1, model, parser)

parallel_runnable = RunnableParallel(
    {
        "joke": RunnablePassthrough(),
        "explaination": RunnableSequence(prompt2, model, parser),
    }
)

final_runnable = RunnableSequence(sequential_runnable, parallel_runnable)

result = final_runnable.invoke({"topic": "Humans"})

print(result)


#                                       -->passthrough runnable --> output
#                                      ^        (pass the joke)
# prompt1 --> llm model --> parser  .. |
#      (generate a joke)               |
#                                       prompt2 --> llm model --> parser
#                                                 (explainatin of joke)
