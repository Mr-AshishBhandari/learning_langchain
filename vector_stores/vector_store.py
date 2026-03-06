from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="myvectors",
    collection_name="sample",
)


d1 = Document(
    page_content="Winning the art competition made her incredibly happy.",
    metadata={"context": "part of story"},
)

d2 = Document(
    page_content="The café on the corner serves really good coffee.",
    metadata={"context": "feedback of cafe"},
)

d3 = Document(
    page_content="The train arrives at the station every hour",
    metadata={"context": "saying of  train master"},
)

d4 = Document(
    page_content="He felt frustrated after losing his wallet.",
    metadata={"context": "every time"},
)

d = [d1, d2, d3, d4]

result = vector_store.add_documents(d)

print(result)
