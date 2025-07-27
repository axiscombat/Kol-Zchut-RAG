from datasets import load_dataset, Dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter


def data_load_huggingface(url, row_count):
    """
    Load a dataset from Hugging Face and return a truncated version without the 'title' column.

    Args:
        url (str): The Hugging Face dataset path or identifier.
        row_count (int): Number of rows to keep from the 'train' split.

    Returns:
        Dataset: A Hugging Face Dataset with the specified number of rows, without the 'title' column.
    """
    ds = load_dataset(url)['train']
    ds_truncated = ds.select(range(row_count))
    return ds_truncated.remove_columns("title")


def chunk_hebrew_news(dataset):
    """
    Chunk Hebrew news articles into overlapping segments for embedding.
    Merges headline, description, and articleBody into one text before chunking.

    Args:
        dataset (Dataset): A Hugging Face Dataset containing Hebrew news articles with
            'headline', 'description', 'articleBody', and 'id' fields.

    Returns:
        Tuple[List[str], Dataset]: A tuple where the first item is a list of texts to embed
            (with Hebrew labels), and the second is a Hugging Face Dataset of chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)

    chunks = []
    for row in dataset:
        article_body = row.get("articleBody", "")
        headline = row.get("headline", "")
        description = row.get("description", "")
        # doc_id = row.get("id", "")

        full_text = f"{headline}\n{description}\n{article_body}".strip()

        if article_body and len(full_text) <= 50000:
            for chunk in text_splitter.split_text(full_text):
                full_chunk_text = f"כותרת: {headline}\nתיאור: {description}\nתוכן: {chunk}"
                chunks.append(full_chunk_text)

    chunked_dataset = Dataset.from_list(chunks)

    texts_to_embed = [
        f"כותרת: {row['headline']}\nתיאור: {row['description']}\nתוכן: {row['text']}"
        for row in chunked_dataset
    ]

    return texts_to_embed, chunked_dataset


def chunk_articles_separately(dataset):
    """
    Chunk Hebrew news articles into overlapping segments, keeping chunks separate per article.

    Args:
        dataset (Dataset): A Hugging Face Dataset containing articles with
            'headline', 'description', 'articleBody', and 'id' fields.

    Returns:
        Dataset: A new Hugging Face Dataset where each row is a chunk belonging to a single article,
        with metadata preserved.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)

    chunks = []
    for row in dataset:
        article_body = row.get("articleBody", "")
        headline = row.get("headline", "")
        description = row.get("description", "")
        doc_id = row.get("id", "")

        if not article_body:
            continue

        full_text = f"כותרת: {headline}\nתיאור: {description}\nתוכן: {article_body}"

        for chunk in text_splitter.split_text(full_text):
            chunks.append({
                "text": chunk,
                "headline": headline,
                "description": description,
                "id": doc_id,
            })

    return Dataset.from_list(chunks)


def chunk_paragraphs_kol_zchut(paragraphs, max_chars=512, overlap=50):
    """
    Chunk long paragraphs from Kol Zchut-style documents into overlapping text segments.

    Args:
        paragraphs (List[Dict]): A list of dictionaries with fields 'content', 'doc_id', 'title', 'link', 'license'.
        max_chars (int, optional): Maximum number of characters per chunk. Defaults to 512.
        overlap (int, optional): Number of overlapping characters between chunks. Defaults to 50.

    Returns:
        List[Dict]: A list of dictionaries, where each contains a content chunk and original metadata.
    """
    chunked = []
    for entry in paragraphs:
        content = entry["content"]
        start = 0
        while start < len(content):
            end = min(start + max_chars, len(content))
            chunk_text = content[start:end]

            chunk = {
                "doc_id": entry["doc_id"],
                "title": entry["title"],
                "link": entry["link"],
                "license": entry["license"],
                "content": chunk_text,
            }
            chunked.append(chunk)

            if end == len(content):
                break
            start += max_chars - overlap

    return chunked
