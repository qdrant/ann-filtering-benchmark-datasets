import json
import os
import random
from typing import List, Dict

import numpy as np
from tqdm import tqdm

from generators.config import DATA_DIR
from generators.generate import DataGenerator


def generate_query(filters: Dict[str, list]):
    fields = list(filters.keys())
    field = random.choice(fields)
    value = random.choice(filters[field])
    return {
        "and": [
            {
                field: {
                    "match": {
                        "value": value
                    }
                }
            }
        ]
    }


def generate_hnm_queries(
        vectors: np.ndarray,
        payloads: List[dict],
        filters: Dict[str, list],
        num_queries: int,
        path
):
    generator = DataGenerator()
    with open(path, "w") as out:
        for _ in tqdm(range(num_queries)):
            ref_id = random.randint(0, len(vectors))
            query_vector = vectors[ref_id]
            query_filter = generate_query(filters=filters)

            closest_ids, best_scores = generator.search(
                vectors=vectors,
                payloads=payloads,
                query=query_vector,
                conditions=query_filter,
            )

            out.write(json.dumps(
                {
                    "query": query_vector.tolist(),
                    "conditions": query_filter,
                    "closest_ids": closest_ids,
                    "closest_scores": best_scores
                }
            ))

            out.write("\n")


def convert_filters(filters):
    res = {}
    for field in filters:
        res[field['name']] = field['values']
    return res


if __name__ == '__main__':
    vectors_path = os.path.join(DATA_DIR, "hnm", "vectors.npy")
    vectors = np.load(vectors_path, allow_pickle=False)

    payloads_path = os.path.join(DATA_DIR, "hnm", "payloads.jsonl")
    payloads = []
    with open(payloads_path) as fd:
        for line in fd:
            payloads.append(json.loads(line))

    filters_path = os.path.join(DATA_DIR, "hnm", "filters.json")
    filters = convert_filters(json.load(open(filters_path)))

    generate_hnm_queries(
        vectors=vectors,
        payloads=payloads,
        filters=filters,
        num_queries=10_000,
        path=os.path.join(DATA_DIR, "hnm", "tests.jsonl")
    )
