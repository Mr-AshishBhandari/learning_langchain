from langchain_classic.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator="\n\n",
)

text = """
The current geopolitical landscape in 2026 is defined by a complex mix of strategic rivalry, shifting alliances, economic competition, and regional conflicts that together shape the global balance of power. At the center of many discussions is the strategic competition between the United States and the China, which spans trade, technology, military influence, and diplomatic alignment across regions such as the Indo-Pacific, Africa, and Latin America. Meanwhile, the ongoing consequences of the Russian invasion of Ukraine continue to reshape European security architecture, strengthening the role of NATO and deepening divisions between the West and Russia. In the Middle East, fragile ceasefires, proxy conflicts, and shifting diplomatic normalization efforts coexist with long-standing tensions involving actors such as Iran, Israel, and the Gulf Cooperation Council. At the same time, the rise of middle powers—countries like India, Turkey, and Brazil—is contributing to a more multipolar international system in which regional influence and strategic autonomy are increasingly important. Global governance institutions such as the United Nations face pressure to adapt to these new realities, while economic blocs including BRICS and the European Union pursue competing visions of economic integration and development. Compounding these dynamics are cross-border challenges—climate change, supply-chain security, energy transitions, cyber competition, and artificial intelligence governance—that blur the line between traditional security and economic policy, forcing states to balance cooperation and rivalry simultaneously. As a result, the modern geopolitical environment is less defined by a single dominant conflict and more by overlapping arenas of competition where diplomacy, economic leverage, military deterrence, and technological leadership collectively determine the trajectory of international relations.
"""

result = splitter.split_text(text=text)

print(result[0])

print(len(result[0]))
