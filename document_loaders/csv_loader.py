from langchain_community.document_loaders import CSVLoader

loader = CSVLoader("document_loaders\\types_of_documents\data.csv")

docs = loader.lazy_load()

for documents in docs:
    print(documents)
