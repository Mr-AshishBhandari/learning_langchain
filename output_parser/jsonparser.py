from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json
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

parser = JsonOutputParser()

prompt = PromptTemplate(
    template="Extract info from review.\n{format_instructions}\n{customer_review}",
    input_variables=["customer_review"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Run Chain
customer_review = "This leaf blower is amazing. It has four settings: candle blower, breeze, windy, tornado."
chain = prompt | model | parser
output = chain.invoke({"customer_review": customer_review})

print(json.dumps(output, indent=4))
