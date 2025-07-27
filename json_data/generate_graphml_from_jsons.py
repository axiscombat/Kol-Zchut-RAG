import json
import os
import networkx as nx
from tqdm import tqdm

# === Config ===
CLEANED_DIR = "../archive_rag_storage/hebrew_news_rag_attempts/article_hebrew_rag_storage_cleaned"
ENT_PATH = os.path.join(CLEANED_DIR, "vdb_entities.json")
REL_PATH = os.path.join(CLEANED_DIR, "vdb_relationships.json")
GRAPHML_PATH = os.path.join(CLEANED_DIR, "graph_chunk_entity_relation.graphml")

# === Load ===
with open(ENT_PATH, "r", encoding="utf-8") as f:
    entities = json.load(f)["data"]

with open(REL_PATH, "r", encoding="utf-8") as f:
    relationships = json.load(f)["data"]

# === Build Graph ===
print("🔧 Building graph...")
G = nx.Graph()

# Add nodes
for entity in tqdm(entities, desc="Adding entities"):
    G.add_node(entity["entity_name"], **{
        # "embedding": entity.get("embedding", []),
        "type": entity.get("entity_type", "unknown")
    })

# Add edges
for rel in tqdm(relationships, desc="Adding relationships"):
    src = rel["src_id"]
    tgt = rel["tgt_id"]
    label = rel.get("relation", "related")

    if src in G.nodes and tgt in G.nodes:
        G.add_edge(src, tgt, label=label)

# === Save ===
print(f"💾 Saving graph to {GRAPHML_PATH}")
nx.write_graphml(G, GRAPHML_PATH)
print("✅ Done.")
