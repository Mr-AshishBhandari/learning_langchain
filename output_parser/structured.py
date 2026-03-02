from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema

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

schema = [
    ResponseSchema(name="fact_1", description="fact 1 of the topic"),
    ResponseSchema(name="fact_2", description="fact 2 of the topic"),
    ResponseSchema(name="fact_3", description="fact 3 of the topic"),
]
parser = StructuredOutputParser(response_schemas=schema)
template = PromptTemplate(
    template="provide 3 facts about the {topic} \n {format_instruction}",
    input_variables=["topic"],
    partial_variables={"format_instruction": parser.get_format_instructions()},
)

chain = template | model | parser

result = chain.invoke({"topic": "Black hole"})

print(result)
