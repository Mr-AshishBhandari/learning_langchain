from langchain_community.document_loaders import DirectoryLoader, TextLoader

directory_loader = DirectoryLoader(
    path="document_loaders",
    glob="**/*.txt",
    loader_cls=TextLoader,
)

# for small size document
# docs = directory_loader.load()

# for large size document
docs = directory_loader.lazy_load()

for document in docs:
    print(document.metadata)
