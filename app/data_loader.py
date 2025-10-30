from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
EMBED_DIM = 384  

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdf(path: str) -> list[str]:
    """Load PDF and split into text chunks."""
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts using local SentenceTransformer."""
    embeddings = embedding_model.encode(texts, convert_to_numpy=True)
    return embeddings.tolist()
