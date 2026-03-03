from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.schema.runnable import RunnableParallel, RunnableSequence

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


prompt1 = PromptTemplate(template="write a tweet on {topic}", input_variables=["topic"])

prompt2 = PromptTemplate(
    template="Write a linkedin post on {topic}", input_variables=["topic"]
)

parser = StrOutputParser()

parallel_runnable = RunnableParallel(
    {
        "tweet": RunnableSequence(prompt1, model, parser),
        "linkedin": RunnableSequence(prompt2, model, parser),
    }
)

result = parallel_runnable.invoke({"topic": "AI"})
print(result)

#     --> prompt1 --> llm model --> parser
#    |       (generate tweet)
# --> |
#    |
#     ---> Prompt2 --> llm model --> parser
#               (generate linkedin post)
