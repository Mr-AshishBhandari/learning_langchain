from langchain_community.document_loaders import TextLoader

textloader = TextLoader("document_loaders\\types_of_documents\sample.txt")
result = textloader.load()
print(result[0])
