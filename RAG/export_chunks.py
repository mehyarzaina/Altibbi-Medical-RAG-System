import json
from RAG.chunking import get_all_chunks

print("Loading chunks...")
chunks = get_all_chunks()

print("Exporting to JSON...")
data = [
    {
        "text": doc.page_content,
        "metadata": doc.metadata
    }
    for doc in chunks
]

with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Saved {len(data)} chunks to chunks.json")