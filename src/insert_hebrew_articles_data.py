import os
import asyncio

from lightrag.prompt import PROMPTS
import prompt_patch

# Override the entity_extraction prompt
PROMPTS["DEFAULT_LANGUAGE"] = prompt_patch.DEFAULT_LANGUAGE
PROMPTS["entity_extraction"] = prompt_patch.EXTRACTION_PROMPT

from lightrag import LightRAG
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import setup_logger, EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

from utils import chunk_articles_separately, data_load_huggingface  # assuming both in utils.py

setup_logger("lightrag", level="INFO")

WORKING_DIR = "../archive_rag_storage/hebrew_news_rag_attempts/article_hebrew_rag_storage_little"
os.makedirs(WORKING_DIR, exist_ok=True)


async def insert(dataset_url: str, row_count: int = 500, batch_size: int = 32):
    """
    Load a hebrew news article dataset from HuggingFace,
    preprocess metadata, and insert entries into a LightRAG vector database.

    Args:
        batch_size (int, optional): The number of entries to insert per batch. Defaults to 32.

    Side Effects:
        - Loads input data from JSON file.
        - Saves chunked data to a new JSON file.
        - Asynchronously inserts data into LightRAG with IDs and metadata.
        - Prints status updates for each batch.
    """
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name='llama3.1:8b',
        llm_model_kwargs={"options": {"num_ctx": 32768}},
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=lambda texts: ollama_embed(texts, embed_model="mxbai-embed-large")
        ),
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()

    # Load dataset from URL and truncate
    dataset = data_load_huggingface(dataset_url, row_count)

    # Chunk articles
    chunked_dataset = chunk_articles_separately(dataset)
    print(f"Total chunks to insert: {len(chunked_dataset)}")

    # Insert in batches asynchronously
    for i in range(0, len(chunked_dataset), batch_size):
        batch = chunked_dataset.select(range(i, min(i + batch_size, len(chunked_dataset))))
        texts = [row["text"] for row in batch]
        await rag.ainsert(texts)
        print(f"Inserted batch {i // batch_size + 1} / {((len(chunked_dataset) - 1) // batch_size) + 1}")

    await rag.finalize_storages()
    print("All chunks inserted and storage finalized.")


if __name__ == "__main__":
    DATASET_URL = "imvladikon/hebrew_news"
    ROW_COUNT = 5

    asyncio.run(insert(DATASET_URL, ROW_COUNT))
