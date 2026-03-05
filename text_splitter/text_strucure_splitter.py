from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
)

# tries to split based on the paragraph (\n\n), if the chunk size exceeds , it tries to split based on line (\n) , then words ( ) and only with character

text = """
Nepal is a small landlocked country in South Asia, located between India and China. It is famous for the Himalayas and Mount Everest, the highest mountain in the world. Nepal has beautiful landscapes, including mountains, hills, forests, and plains.

Nepal is also rich in culture and traditions. People celebrate festivals like Dashain and Tihar, and many religions such as Hinduism and Buddhism are practiced. The capital city, Kathmandu, is known for its historic temples and heritage sites. 🇳🇵
"""

result = splitter.split_text(text=text)

print(result)
print(len(result))
