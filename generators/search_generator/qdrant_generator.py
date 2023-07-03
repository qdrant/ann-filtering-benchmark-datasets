from time import sleep
from typing import Callable, Iterable, Tuple

import numpy as np
from qdrant_client import QdrantClient, models


# Generates reference search result by applying exact search
def index_qdrant(embeddings: np.ndarray, payload: list):
    """
    Requires running qdrant server on localhost:6333:

    docker run --rm -it --network=host qdrant/qdrant:latest

    """
    client = QdrantClient(prefer_grpc=True)

    client.recreate_collection(
        collection_name="tmp",
        vectors_config=models.VectorParams(
            size=embeddings.shape[1],
            distance=models.Distance.COSINE,
        ),
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=0,
        )
    )

    client.upload_collection(
        collection_name="tmp",
        vectors=embeddings,
        payload=payload,
        batch_size=100,
    )

    sleep(1)


def search_qdrant(
        sample_embeddings: np.ndarray,
        filter_generator: Callable[[], Tuple[dict, dict]],
        n: int,
        top: int
) -> Iterable[dict]:
    client = QdrantClient(prefer_grpc=True)

    for _ in range(n):
        query_vector = sample_embeddings[np.random.randint(sample_embeddings.shape[0])]
        dataset_query, qdrant_query = filter_generator()
        hits = client.search(
            collection_name="tmp",
            query_vector=query_vector,
            query_filter=models.Filter(**qdrant_query),
            limit=top,
            search_params=models.SearchParams(
                exact=True,
            )
        )
        yield {
            "query": query_vector.tolist(),
            "conditions": dataset_query,
            "closest_ids": [hit.id for hit in hits],
            "closest_scores": [hit.score for hit in hits]
        }
