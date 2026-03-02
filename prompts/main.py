from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
import streamlit as st

from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")


llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
    huggingfacehub_api_token=api_key,
)

model = ChatHuggingFace(llm=llm)

template = PromptTemplate(
    template="""
Explain about the ohms law {theory} in mathematical terms.
""",
    input_variables=["theory"],
)

st.title("Prompts in langchain.")

user_prompt = st.text_input("Enter theory ")
prompt = template.invoke({"theory": user_prompt})


if st.button("Submit"):
    result = model.invoke(prompt)
    st.write(result.content)
