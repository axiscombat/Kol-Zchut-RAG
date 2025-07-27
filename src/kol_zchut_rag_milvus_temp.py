import json
import os
import asyncio
from tqdm import tqdm
from lightrag.prompt import PROMPTS

import prompt_patch

# Override the entity_extraction prompt
PROMPTS["DEFAULT_LANGUAGE"] = prompt_patch.DEFAULT_LANGUAGE
PROMPTS["entity_extraction"] = prompt_patch.EXTRACTION_PROMPT

from lightrag import LightRAG
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import setup_logger, EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

from utils import chunk_paragraphs_kol_zchut  # assuming both in utils.py

setup_logger("lightrag-kol_zchut-LOGGER", level="INFO")

WORKING_DIR = "kol_zchut_rag_storage_1000_v3_milvus"
os.makedirs(WORKING_DIR, exist_ok=True)


def clean_text(text: str) -> str:
    """
    Clean a text string by stripping leading/trailing spaces, removing newlines,
    double quotes, and backslashes. This is done to make links working properly.

    Args:
        text (str): The input string to clean.

    Returns:
        str: A cleaned string with unnecessary characters removed.
    """
    return text.strip().replace('\n', ' ').replace('"', '').replace('\\', '')


async def insert(batch_size: int = 32):
    """
    Load a chunked Hebrew paragraph corpus, preprocess metadata, and insert entries into a LightRAG vector database.

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
            func=lambda texts: ollama_embed(texts, embed_model="mxbai-embed-large"),
        ),
        vector_storage="MilvusVectorDBStorage",
        vector_db_storage_cls_kwargs={
            "host": "172.25.109.227",
            "port": 19530,
            "collection_name": "kol_zchut_embeddings",
            "dim": 1024,
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "drop_old": True
        }
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()

    # Assuming 'sampled' is your list of 1000 entries
    with open("Webiks_Hebrew_RAGbot_KolZchut_Paragraphs_Corpus_SMALL.json", "r", encoding="utf-8") as f:
        sampled = json.load(f)

    chunked_dataset = chunk_paragraphs_kol_zchut(sampled, max_chars=512, overlap=50)

    with open("Webiks_Hebrew_RAGbot_KolZchut_Paragraphs_Chunked.json", "w", encoding="utf-8") as f:
        json.dump(chunked_dataset, f, ensure_ascii=False, indent=2)

    tqdm.write(f"✅ Created {len(chunked_dataset)} chunked entries from {len(sampled)} original paragraphs")

    # Insert in batches asynchronously
    for i in tqdm(range(0, len(chunked_dataset), batch_size), desc="📦 Inserting batches"):
        batch = chunked_dataset[i: i + batch_size]
        inputs = []
        file_paths = []
        ids = []

        for j, row in enumerate(tqdm(batch, desc=f"  🔹 Batch {i // batch_size + 1}", leave=False)):
            content = row["content"]
            doc_id = str(row["doc_id"])  # ensure it's a string
            title = clean_text(row["title"])
            link = clean_text(row["link"])

            inputs.append(content)
            file_paths.append(f"{title} - {link}")
            ids.append(f"{doc_id}_{j}")  # ensures uniqueness

        await rag.ainsert(
            input=inputs,
            file_paths=file_paths,
            ids=ids
        )

        tqdm.write(f"✅ Inserted batch {i // batch_size + 1} / {((len(chunked_dataset) - 1) // batch_size) + 1}")

    await rag.finalize_storages()
    tqdm.write("🎉 All chunks inserted and storage finalized.")


if __name__ == "__main__":
    # ROW_COUNT = 1000

    asyncio.run(insert())
