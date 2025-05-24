import json
import os
import tqdm

from generators.config import DATA_DIR

from .prepare_data import client
from .prepare_data import read_data, DATASETS, QDRANT_COLLECTION_NAME, LIMIT, EXACT_QUERY_COUNT
from qdrant_client import models


def run_exact_search(output_file: str = "search_result_embeddings.jsonl"):
    """
    Runs exact search against the Qdrant collection.
    Saves the result in vector-db-benchmark compatible format.
    """
    points = read_data(DATASETS, limit=EXACT_QUERY_COUNT)

    print(f"Saving to {output_file}...")

    batch = []

    for point in tqdm.tqdm(points, desc="Running exact search"):
        vector = point.vector
        batch.append(vector)


    responses = client.query_batch_points(
        collection_name=QDRANT_COLLECTION_NAME,
        requests=[
            models.QueryRequest(
                query=vector,
                limit=LIMIT,
                params=models.SearchParams(exact=True),
                with_payload=["id"]
            )
            for vector in batch
        ],
        timeout=3600
    )

    responses_approx = client.query_batch_points(
        collection_name=QDRANT_COLLECTION_NAME,
        requests=[
            models.QueryRequest(
                query=vector,
                limit=LIMIT,
                params=models.SearchParams(exact=False),
                with_payload=["id"]
            )
            for vector in batch
        ],
        timeout=3600
    )

    correct = 0
    total = 0
    for appx, exact in zip(responses_approx, responses):
        ids_appx = set(hit.payload["id"] for hit in appx.points )
        ids_exact = set(hit.payload["id"] for hit in exact.points)

        correct += len(ids_appx & ids_exact)
        total += len(ids_appx)

    print(f"Accuracy: {correct / total}")


    with open(output_file, "w", encoding="utf-8") as output_f:
        for vector, response in zip(batch, responses):
            hits = response.points

            record = {
                "query": vector,
                "closest_ids": [hit.payload["id"] - EXACT_QUERY_COUNT for hit in hits],
                "closest_scores": [hit.score for hit in hits],
                "conditions": None
            }

            output_f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    dataset_file = os.path.join(DATA_DIR, "cohere_wiki", "tests.jsonl")
    run_exact_search(dataset_file)