import json
import os
from multiprocessing import Pool

import numpy as np
import tqdm

from generators.config import DATA_DIR
from generators.random_data.generate import DataGenerator


def generate_conditions(seed, condition_generator):
    if seed % 3 == 0:
        # Single condition
        return {
            "and": [
                {
                    'a': condition_generator()
                }
            ]
        }

    if seed % 3 == 1:
        # Double "and" condition
        return {
            "and": [
                {
                    "a": condition_generator()
                },
                {
                    "b": condition_generator()
                }
            ]
        }

    if seed % 3 == 2:
        # Double "or" condition
        return {
            "or": [
                {
                    "a": condition_generator()
                },
                {
                    "a": condition_generator()
                }
            ]
        }


def generate_samples(generator, num_queries, dim, vectors, payloads, path, condition_generator, top=25):
    with open(path, "w") as out:
        for i in tqdm.tqdm(range(num_queries)):
            query = generator.random_vectors(1, dim=dim)[0]
            conditions = generate_conditions(seed=i, condition_generator=condition_generator)

            closest_ids, best_scores = generator.search(
                vectors=vectors,
                payloads=payloads,
                query=query,
                conditions=conditions,
                top=top
            )

            out.write(json.dumps(
                {
                    "query": query.tolist(),
                    "conditions": conditions,
                    "closest_ids": closest_ids,
                    "closest_scores": best_scores
                }
            ))

            out.write("\n")


def generate_data(
        generator,
        size,
        dim,
        path,
        num_queries,
        payload_gen,
        condition_gen
):
    os.makedirs(path, exist_ok=True)
    vectors = generator.random_vectors(size, dim)

    np.save(os.path.join(path, "vectors.npy"), vectors, allow_pickle=False)

    payloads = [payload_gen() for _ in range(size)]

    with open(os.path.join(path, "payloads.jsonl"), "w") as out:
        for payload in payloads:
            out.write(json.dumps(payload))
            out.write("\n")

    generate_samples(
        generator=generator,
        num_queries=num_queries,
        dim=dim,
        vectors=vectors,
        payloads=payloads,
        path=os.path.join(path, "tests.jsonl"),
        condition_generator=condition_gen
    )


if __name__ == '__main__':
    generator = DataGenerator(vocab_size=1000)

    generate_data(
        generator=generator,
        size=1_000_000,
        dim=100,
        path=os.path.join(DATA_DIR, "random_keywords_test"),
        num_queries=10_000,
        payload_gen=lambda: {
            "a": generator.sample_keyword(),
            "b": generator.sample_keyword()
        },
        condition_gen=generator.random_match_keyword
    )
