from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.schema.runnable import (
    RunnableParallel,
    RunnableSequence,
    RunnablePassthrough,
    RunnableLambda,
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
    template="Generate a joke on {topic}", input_variables=["topic"]
)

parser = StrOutputParser()

joke_generate_runnable = RunnableSequence(prompt1, model, parser)

parallel_runnable = RunnableParallel(
    {
        "joke": RunnablePassthrough(),
        "word_count": RunnableLambda(lambda x: x.count(" ") + 1),
    }
)

runnable = RunnableSequence(joke_generate_runnable, parallel_runnable)

result = runnable.invoke({"topic": "Humans"})

print(result)


#                                       -->passthrough runnable --> output
#                                      ^        (pass the joke)
# prompt1 --> llm model --> parser  .. |
#      (generate a joke)               |
#                                       -->       lambda runnable
#                                          (count no.of words in generated joke)
