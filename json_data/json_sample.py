import json

INPUT_FILE = "Webiks_Hebrew_RAGbot_KolZchut_Paragraphs_Corpus_v1.0.json"
OUTPUT_FILE = "../src/Webiks_Hebrew_RAGbot_KolZchut_Paragraphs_Corpus_SMALL.json"
SAMPLE_SIZE = 1000

# Load original column-based data
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Extract all row indices
all_ids = list(raw_data["doc_id"].keys())
print(f"Total rows in full corpus: {len(all_ids)}")

# Convert to list of dicts
paragraphs = []
for i in all_ids:
    paragraph = {
        "doc_id": raw_data["doc_id"][i],
        "title": raw_data["title"][i],
        "content": raw_data["content"][i],
        "link": raw_data["link"][i],
        "license": raw_data["license"],
    }
    paragraphs.append(paragraph)

# Take first SAMPLE_SIZE entries
sampled = paragraphs[:SAMPLE_SIZE]

# Save
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(sampled, f, ensure_ascii=False, indent=2)

print(f"✅ Saved first {len(sampled)} paragraphs to {OUTPUT_FILE}")
