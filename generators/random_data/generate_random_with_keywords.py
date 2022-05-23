import json
import os

import numpy as np

from generators.config import DATA_DIR
from generators.random_data.generate import DataGenerator


def generate_with_keywords(size, dim, path, num_queries):
    os.makedirs(path, exist_ok=True)

    generator = DataGenerator(vocab_size=1000)
    vectors = generator.random_vectors(size, dim)

    np.save(os.path.join(path, "vectors.npy"), vectors, allow_pickle=False)

    payloads = [
        {
            "a": generator.sample_keyword(),
            "b": generator.sample_keyword()
        }
        for _ in range(size)
    ]

    with open(os.path.join(path, "payloads.jsonl"), "w") as out:
        for payload in payloads:
            out.write(json.dumps(payload))
            out.write("\n")

    with open(os.path.join(path, "tests.jsonl"), "w") as out:
        for i in range(num_queries):
            query = generator.random_vectors(1, dim=dim)[0]
            conditions = {}
            if i % 3 == 0:
                # Single condition
                conditions = {
                    "and": [
                        generator.random_match_keyword()
                    ]
                }

            if i % 3 == 1:
                # Double "and" condition
                conditions = {
                    "and": [
                        generator.random_match_keyword(),
                        generator.random_match_keyword()
                    ]
                }

            if i % 3 == 2:
                # Double "or" condition
                conditions = {
                    "or": [
                        generator.random_match_keyword(),
                        generator.random_match_keyword()
                    ]
                }

            closest_ids, best_scores = generator.search(
                vectors=vectors,
                payloads=payloads,
                query=query,
                conditions=conditions,
                top=25
            )

            out.write(json.dumps(
                {
                    "query": query,
                    "conditions": conditions,
                    "closest_ids": closest_ids,
                    "closest_scores": best_scores
                }
            ))

            out.write("\n")


if __name__ == '__main__':
    generate_with_keywords(
        size=1_000_000,
        dim=100,
        path=os.path.join(DATA_DIR, "random_keywords_1M"),
        num_queries=10000
    )

    # generate_with_keywords(
    #     size=100_000,
    #     dim=2048,
    #     path=os.path.join(DATA_DIR, "random_keywords_100k"),
    #     num_queries=10000
    # )

