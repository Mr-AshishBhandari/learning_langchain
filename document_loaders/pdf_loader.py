from langchain_community.document_loaders import PyPDFLoader

pdf_loader = PyPDFLoader("document_loaders\\types_of_documents\sample.pdf")

contents = pdf_loader.load()

print(len(contents)) # no.of pages in pdf 

print(contents[0])

print(type(contents))
