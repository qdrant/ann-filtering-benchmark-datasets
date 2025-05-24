import json
import os
from typing import Iterable

import tqdm
import numpy as np

from generators.config import DATA_DIR

from .hf import read_dataset_stream
from qdrant_client import QdrantClient, models

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
QDRANT_CLUSTER_URL = os.getenv("QDRANT_CLUSTER_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
QDRANT_COLLECTION_NAME = "benchmark"
EXACT_QUERY_COUNT = 1000
LIMIT=100
LIMIT_POINTS = 100_000
DATASETS = [
    "Cohere/wikipedia-22-12-en-embeddings",
    "Cohere/wikipedia-22-12-simple-embeddings",
    "Cohere/wikipedia-22-12-de-embeddings"
]


client = QdrantClient(
    url=QDRANT_CLUSTER_URL,
    api_key=QDRANT_API_KEY,
    prefer_grpc=True,
    timeout=3600 # For full-scan search
)


def create_collection(force_recreate=False):
    if force_recreate:
        client.delete_collection(QDRANT_COLLECTION_NAME)

    if client.collection_exists(QDRANT_COLLECTION_NAME):
        return

    client.create_collection(
        QDRANT_COLLECTION_NAME,
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                always_ram=True,
                quantile=0.99,
            )
        ),
        hnsw_config=models.HnswConfigDiff(
            m=0,
        ),
        vectors_config=models.VectorParams(
            size=768,
            distance=models.Distance.COSINE,
            on_disk=True,
            datatype=models.Datatype.FLOAT32,
        ),
        optimizers_config=models.OptimizersConfigDiff(
            max_segment_size=50_000_000,
            default_segment_number=1,
        )
    )


def read_data(
        datasets: list[str],
        skip_first: int = 0,
        limit: int = LIMIT_POINTS
) -> Iterable[models.PointStruct]:
    
    n = 0
    for dataset in datasets:
        stream = read_dataset_stream(dataset, split="train")
        for item in stream:
            n += 1

            if n <= skip_first:
                continue

            if n >= limit:   
                return

            embedding = item.pop("emb")

            yield models.PointStruct(
                id=n,
                vector=embedding.tolist(),
                payload=item,
            )


def load_all():

    # Use first 1000 points for testing
    skip_first = EXACT_QUERY_COUNT

    points = list(read_data(DATASETS, skip_first=skip_first, limit=LIMIT_POINTS + skip_first))

    vectors = np.stack([np.array(point.vector) for point in points]).astype(np.float32)

    print("Vectors shape:", vectors.shape)

    DATASET_DIR = os.path.join(DATA_DIR, "cohere_wiki")

    os.makedirs(DATASET_DIR, exist_ok=True)

    np.save(os.path.join(DATASET_DIR, "vectors.npy"), vectors)

    payloads_path = os.path.join(DATASET_DIR, "payloads.jsonl")
    with open(payloads_path, "w") as f:
        for point in points:
            f.write(json.dumps(point.payload) + "\n")


    client.upload_points(
        collection_name=QDRANT_COLLECTION_NAME,
        points=tqdm.tqdm(points, desc="Uploading points"),
        parallel=8,
        batch_size=64,
    )

def main():
    create_collection(force_recreate=True)
    load_all()


if __name__ == "__main__":
    main()
