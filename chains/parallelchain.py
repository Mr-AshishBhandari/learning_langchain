from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

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
    template="explain the about the {topic} in brief",
    input_variables=["topic"],
)
template2 = PromptTemplate(
    template="Make a quiz questions about the {topic}.", input_variables=["topic"]
)
template3 = PromptTemplate(
    template="Merge the text and the quiz in a same document. text -> {text} quiz -> {quiz}",
    input_variables=["text", "quiz"],
)
parser = StrOutputParser()

# parallel chain
parellel_chain = RunnableParallel(
    {
        "text": template1 | model | parser,
        "quiz": template2 | model | parser,
    }
)

merge_Chain = template3 | model | parser
chain = parellel_chain | merge_Chain

result = chain.invoke({"topic": "Black hole"})
print(result)
