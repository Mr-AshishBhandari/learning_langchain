from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_classic.schema.runnable import (
    RunnableBranch,
    RunnableSequence,
    RunnableLambda,
)

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


class Sentiment(BaseModel):
    sentiment: Literal["positive", "negative"] = Field("Sentiment of feedback")


sentiment_parser = PydanticOutputParser(pydantic_object=Sentiment)
parser = StrOutputParser()

prompt = PromptTemplate(
    template="Based on the feedback, find the sentiment of feedback \n {feedback} {format_instruction}",
    input_variables=["feedback"],
    partial_variables={
        "format_instruction": sentiment_parser.get_format_instructions()
    },
)

prompt1 = PromptTemplate(
    template="Generate response for the positive feedback {feedback}",
    input_variables=["feedback"],
)

prompt2 = PromptTemplate(
    template="Generate response for the negative feedback {feedback}",
    input_variables=["feedback"],
)

sentiment_runnable = RunnableSequence(prompt, model, sentiment_parser)

branch_runnable = RunnableBranch(
    (lambda x: x.sentiment == "positive", RunnableSequence(prompt1, model, parser)),
    (lambda x: x.sentiment == "negative", RunnableSequence(prompt2, model, parser)),
    RunnableLambda(lambda x: "Can't find the sentiment"),
)

runnable = RunnableSequence(sentiment_runnable, branch_runnable)

result = runnable.invoke(
    {"feedback": "This is a nice product and is very convinent to use."}
)

print(result)


#                                            (response for positive feedback)
#                                        ---> prompt2 --> llm --> parser
#                                       |
#                                    (positive feedback)
#                                       |
# Prompt --> llm --> pydantic_parser ----|
#   (sentiment from feedback)           |
#                                   (negative feedback)
#                                       |
#                                       ---> prompt2 --> llm --> parser
#                                               (response for negative feedback)
#
