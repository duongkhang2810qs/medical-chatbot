# build_index.py
from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer

from config import (
    CBC_KB_PATH,
    BIOCHEM_KB_PATH,
    LAB_KB_PATH,
    COLLECTION_NAME,
    QDRANT_HOST,
    QDRANT_PORT,
    EMBEDDING_MODEL_NAME,
)
from lab_core import (
    load_json,
    save_json,
    normalize_kb_item,
    build_embedding_text,
)


BATCH_SIZE = 512


def load_kb_file(path, panel: str) -> list[dict]:
    if not path.exists():
        print(f"Warning: missing KB file: {path}")
        return []

    data = load_json(path)

    if not isinstance(data, list):
        raise ValueError(f"KB file must be a JSON list: {path}")

    normalized = []

    for idx, item in enumerate(data):
        normalized.append(normalize_kb_item(item, panel=panel, idx=idx))

    return normalized


def merge_kb() -> list[dict]:
    print("=" * 80)
    print("MERGE CBC + BIOCHEM KB")
    print("=" * 80)

    cbc_kb = load_kb_file(CBC_KB_PATH, panel="CBC")
    biochem_kb = load_kb_file(BIOCHEM_KB_PATH, panel="BIOCHEM")

    lab_kb = cbc_kb + biochem_kb

    for item in lab_kb:
        item["embedding_text"] = item.get("embedding_text") or build_embedding_text(item)

    save_json(LAB_KB_PATH, lab_kb)

    print(f"CBC chunks: {len(cbc_kb)}")
    print(f"BIOCHEM chunks: {len(biochem_kb)}")
    print(f"LAB chunks: {len(lab_kb)}")
    print(f"Saved merged LAB KB: {LAB_KB_PATH}")

    return lab_kb


def recreate_collection(client: QdrantClient, vector_dim: int) -> None:
    print(f"Recreating Qdrant collection: {COLLECTION_NAME}")

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=vector_dim,
            distance=Distance.COSINE,
        ),
    )


def upload_in_batches(client: QdrantClient, embeddings, kb: list[dict]) -> None:
    total = len(kb)

    print(f"Uploading {total} vectors to Qdrant...")
    print(f"Batch size: {BATCH_SIZE}")

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)

        points = []

        for idx in range(start, end):
            points.append({
                "id": idx,
                "vector": embeddings[idx].tolist(),
                "payload": kb[idx],
            })

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )

        print(f"Uploaded {end}/{total}")


def main():
    print("=" * 80)
    print("BUILD UNIFIED QDRANT INDEX")
    print("=" * 80)

    kb = merge_kb()

    if not kb:
        print("KB is empty. Stop.")
        return

    print("\nLoading embedding model...")
    print(f"Model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = [item.get("embedding_text", "") for item in kb]

    print(f"\nEncoding {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=64,
        normalize_embeddings=True,
    )

    vector_dim = len(embeddings[0])
    print(f"Vector dimension: {vector_dim}")

    print(f"\nConnecting Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    recreate_collection(client, vector_dim)
    upload_in_batches(client, embeddings, kb)

    print("\nDONE")
    print(f"Collection: {COLLECTION_NAME}")
    print("Next step:")
    print("python build_graph.py")


if __name__ == "__main__":
    main()