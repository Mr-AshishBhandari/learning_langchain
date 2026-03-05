from langchain_classic.text_splitter import PythonCodeTextSplitter

splitter = PythonCodeTextSplitter(
    chunk_size=200,
)

code = """
class Animal:
    def __init__(self, name):
        self.name = name

    def move(self):
        return f'{self.name} moved '

    dog = Animal('doggy')
    value = dog.move()
    print(value)
"""
result = splitter.split_text(text=code)

print(result)
