from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda

from pydantic import BaseModel, Field
from typing import Literal

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


# pydantic model for validation
class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Sentiment of the feedback"
    )


pydantic_parser = PydanticOutputParser(pydantic_object=Feedback)  # for sentiment output
string_parser = StrOutputParser()

feedback_prompt = PromptTemplate(
    template="Analyze the sentiment of the Feedback \n {feedback} \n {format_instruction}",
    input_variables=["feedback"],
    partial_variables={"format_instruction": pydantic_parser.get_format_instructions()},
)


positive_feedback_prompt = PromptTemplate(
    template="Provide an appropriate response for the positive feedback . \n {feedback}",
    input_variables=["feedback"],
)

negative_feedback_prompt = PromptTemplate(
    template="Provide an appropriate response for the negative feedback . \n {feedback}",
    input_variables=["feedback"],
)

sentiment_chain = feedback_prompt | model | pydantic_parser


conditional_chain = RunnableBranch(
    (
        lambda x: x.sentiment == "positive",
        positive_feedback_prompt | model | string_parser,
    ),
    (
        lambda x: x.sentiment == "negative",
        negative_feedback_prompt | model | string_parser,
    ),
    RunnableLambda(lambda x: "Can't analyze the sentiment"),
)
chain = sentiment_chain | conditional_chain

result = chain.invoke({"feedback": "This is the terrible item i have seen in my life"})

print(result)
