import asyncio
import inspect
from pathlib import Path

from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status


async def print_stream(stream):
    """
    Print asynchronous stream chunks as they are received.

    Args:
        stream (AsyncGenerator): An async generator yielding string chunks from a streamed LLM response.

    Returns:
        None
    """
    async for chunk in stream:
        print(chunk, end="", flush=True)


async def initialize_rag(WORKING_DIR: str):
    """
    Initialize a LightRAG instance with specified configuration and prepare storage.

    Args:
        WORKING_DIR (str): Path to the directory where LightRAG will store data and models.

    Returns:
        LightRAG: An initialized LightRAG instance ready for querying or inserting.
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

    return rag


async def query(rag: LightRAG, query_str: str, query_mode: str) -> None:
    """
    Perform multiple search queries (naive, local, global, hybrid) using the given LightRAG instance.

    Args:
        rag (LightRAG): An initialized LightRAG instance to query.
        query_str (str): The user's question or query in Hebrew.
        query_mode (str): The search mode to demonstrate (not currently used—defaults to all modes for demo).

    Returns:
        None
    """
    await rag.initialize_storages()
    await initialize_pipeline_status()

    for mode in ["naive", "local", "global", "hybrid"]:
        print("\n=====================")
        print(f"Query mode: {mode}")
        print("=====================")

        resp = await rag.aquery(
            query=query_str,
            param=QueryParam(mode=mode, stream=True),
        )
        if inspect.isasyncgen(resp):
            await print_stream(resp)
        else:
            print(resp)


async def main(query_prompt):
    """
    Entry point for running an example RAG query using LightRAG.
    Initializes a RAG pipeline and runs a sample query against it using different search modes.

    :param query_prompt:
    :return:  None
    """
    # Create path relative to this script's location
    script_dir = Path(__file__).parent
    WORKING_DIR = script_dir / "kol_zchut_rag_storage_1000_v3"
    rag = await initialize_rag(str(WORKING_DIR))


    await query(rag, query_prompt, query_mode="hybrid")  # query_mode is passed but ignored in function


if __name__ == "__main__":
    query_prompt = "איזה מסמכים או פרטים משפחת האומנה צריכה לקבל כשהיא מקבלת ילד לאומנה?"
    asyncio.run(main(query_prompt))
