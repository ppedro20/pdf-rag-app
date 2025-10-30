import asyncio
from concurrent.futures import ThreadPoolExecutor
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

executor = ThreadPoolExecutor()

class QdrantStorage:
    def __init__(self, url="http://localhost:6333", collection="docs", dim=384):
        """
        Async-friendly Qdrant wrapper.
        dim: embedding dimension (e.g., 384 for MiniLM embeddings)
        """
        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection
        self.dim = dim

        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    async def upsert(self, ids, vectors, payloads):
        """Upsert points asynchronously."""
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, lambda: self.client.upsert(self.collection, points=points))

    async def search(self, query_vector, top_k: int = 5):
        """Search asynchronously."""
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            executor,
            lambda: self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                with_payload=True,
                limit=top_k
            )
        )

        contexts = []
        sources = set()
        for r in results:
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")
            if text:
                contexts.append(text)
                sources.add(source)

        return {"contexts": contexts, "sources": list(sources)}
