import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid
import os
import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
from app.data_loader import load_and_chunk_pdf
from app.vector_db import QdrantStorage
from app.custom_types import RAGQueryResult, RAGSearchResult, RAGUpsertResult, RAGChunkAndSrc

from sentence_transformers import SentenceTransformer
from transformers import pipeline

load_dotenv()

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-large", device=-1)

executor = ThreadPoolExecutor()

async def embed_texts_async(texts):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, lambda: embedding_model.encode(texts, convert_to_numpy=True).tolist())

async def run_qa_async(prompt):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, lambda: qa_pipeline(prompt, max_length=512, do_sample=False))
    return result[0]['generated_text'].strip()

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
    throttle=inngest.Throttle(
        limit=2, period=datetime.timedelta(minutes=1)
    ),
    rate_limit=inngest.RateLimit(
        limit=1,
        period=datetime.timedelta(hours=4),
        key="event.data.source_id",
    ),
)
async def rag_ingest_pdf(ctx: inngest.Context):
    async def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        ctx.logger.info(f"Loading PDF from: {pdf_path}")
        chunks = await asyncio.get_event_loop().run_in_executor(executor, lambda: load_and_chunk_pdf(pdf_path))
        ctx.logger.info(f"Loaded {len(chunks)} chunks from PDF")
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    async def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        ctx.logger.info(f"Embedding {len(chunks)} chunks...")
        vecs = await embed_texts_async(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        ctx.logger.info(f"Upserting {len(ids)} vectors to Qdrant...")
        
        store = QdrantStorage()
        await store.upsert(ids, vecs, payloads)
        
        ctx.logger.info(f"Successfully ingested {len(chunks)} chunks")
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    return ingested.model_dump()

@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    async def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        ctx.logger.info(f"Searching for question: '{question}' with top_k={top_k}")
        query_vec = (await embed_texts_async([question]))[0]
        ctx.logger.info(f"Query vector generated, length: {len(query_vec)}")
        
        store = QdrantStorage()
        
        # Check collection stats
        try:
            loop = asyncio.get_event_loop()
            collection_info = await loop.run_in_executor(
                executor, 
                lambda: store.client.get_collection(store.collection)
            )
            ctx.logger.info(f"Collection '{store.collection}' has {collection_info.points_count} points")
        except Exception as e:
            ctx.logger.error(f"Failed to get collection info: {e}")
        
        found = await store.search(query_vec, top_k)
        ctx.logger.info(f"Search returned {len(found['contexts'])} contexts from sources: {found['sources']}")
        
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k), output_type=RAGSearchResult)

    if not found.contexts:
        ctx.logger.warning("No contexts found! The vector DB might be empty or the query doesn't match any documents.")
        return {"answer": "No relevant information found in the documents.", "sources": [], "num_contexts": 0}

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )

    answer = await run_qa_async(user_content)
    return {"answer": answer, "sources": found.sources, "num_contexts": len(found.contexts)}

app = FastAPI()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        store = QdrantStorage()
        info = store.client.get_collection("docs")
        
        return {
            "status": "healthy",
            "qdrant": "connected",
            "points": info.points_count,
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])