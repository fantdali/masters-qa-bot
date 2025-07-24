import os
import json
from pathlib import Path


def read_blocks(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    return blocks


def build_corpus(data_dir, keys):
    corpus = []
    for key in keys:
        for source in ["page", "plan"]:
            txt_path = Path(data_dir) / f"{key}_{source}.txt"
            if not txt_path.exists():
                continue
            blocks = read_blocks(txt_path)
            for i, block in enumerate(blocks):
                corpus.append(
                    {
                        "program": key,
                        "source": source,
                        "block_id": f"{key}_{source}_{i}",
                        "text": block,
                    }
                )
    return corpus


def main():
    data_dir = "./data"
    keys = ["ai", "ai_product"]
    corpus = build_corpus(data_dir, keys)
    with open(os.path.join(data_dir, "corpus.jsonl"), "w", encoding="utf-8") as f:
        for doc in corpus:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"Corpus saved: {len(corpus)} blocks")


if __name__ == "__main__":
    main()
