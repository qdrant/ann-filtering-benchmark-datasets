import json
import os
import random
import string
from typing import List

import numpy as np
import tqdm
from haversine import haversine
from sklearn.metrics.pairwise import cosine_similarity


class DataGenerator:

    def __init__(self, vocab_size=1000):
        self.vocab = [self.random_keyword() for _ in range(vocab_size)]

    def random_keyword(self):
        letters = string.ascii_letters
        return "".join(random.sample(letters, 5))

    def sample_keyword(self):
        return random.choice(self.vocab)

    def random_float(self):
        return random.random()

    def random_int(self, rng=100):
        return random.randint(0, rng)

    def random_geo(self):
        return {
            "lon": random.uniform(-180.0, 180.0),
            "lat": random.uniform(-90.0, 90.0)
        }

    def random_range_query(self):
        a, b = random.random(), random.random()
        return {
            "range": {
                "gt": min(a, b),
                "lt": max(a, b)
            }
        }

    def random_match_keyword(self):
        return {
            "match": {
                "value": self.sample_keyword()
            }
        }

    def random_match_int(self, rng=100):
        return {
            "match": {
                "value": self.random_int(rng)
            }
        }

    def random_geo_query(self, radius=10_000):
        return {
            "geo": {
                **self.random_geo(),
                "radius": random.randint(1000, radius)
            }
        }

    def random_vectors(self, size, dim):
        return np.random.rand(size, dim).astype(np.float32)

    def check_range(self, value, condition: dict):
        return condition['gt'] < value < condition['lt']

    def check_match(self, value, condition: dict):
        return (
            condition['value'] in value if isinstance(value, list) else value == condition['value']
        )

    def check_geo(self, value, condition: dict):
        a = (value['lat'], value['lon'])
        b = (condition['lat'], condition['lon'])
        return haversine(a, b) * 1000 < condition['radius']

    def check_condition(self, value, condition):
        if 'match' in condition:
            return self.check_match(value, condition['match'])
        if 'range' in condition:
            return self.check_range(value, condition['range'])
        if 'geo' in condition:
            return self.check_geo(value, condition['geo'])
        raise ValueError(f"Unknown condition: {condition}")

    def check_conditions(self, payload: dict, conditions: dict):
        if 'and' in conditions:
            and_res = True
            for field_condition in conditions['and']:
                for field, condition in field_condition.items():
                    and_res = and_res and self.check_condition(value=payload[field], condition=condition)
            return and_res

        if 'or' in conditions:
            or_res = False
            for field_condition in conditions['or']:
                for field, condition in field_condition.items():
                    or_res = or_res or self.check_condition(value=payload[field], condition=condition)
            return or_res

        raise ValueError(f"Unknown conditions: {conditions}")

    def search(
            self,
            vectors: np.ndarray,
            payloads: List[dict],
            query: np.ndarray,
            conditions: dict = None,
            top=25):

        mask = np.array([self.check_conditions(payload, conditions) for payload in payloads])

        # Select only matched by payload vectors
        filtered_vectors = vectors[mask]
        # List of original ids
        raw_ids = np.arange(0, len(vectors))
        # List of ids, filtered by payload
        filtered_ids = raw_ids[mask]
        if len(filtered_vectors) == 0:
            return [], []
        # Scores among filtered vectors
        scores = cosine_similarity([query], filtered_vectors)[0]
        # Ids in filtered matrix
        top_scores_ids = np.argsort(scores)[-top:][::-1]
        top_scores = scores[top_scores_ids]
        # Original ids before filtering
        original_ids = filtered_ids[top_scores_ids]
        return list(map(int, original_ids)), list(map(float, top_scores))


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


def generate_samples(
        generator: DataGenerator,
        num_queries,
        dim,
        vectors,
        payloads,
        path,
        condition_generator,
        top=25,
):
    with open(path, "w") as out:
        for i in tqdm.tqdm(range(num_queries)):
            query = generator.random_vectors(1, dim=dim)[0]
            conditions = generate_conditions(seed=i, condition_generator=condition_generator)

            closest_ids, best_scores = generator.search(
                vectors=vectors,
                payloads=payloads,
                query=query,
                conditions=conditions,
                top=top,
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


def generate_random_dataset(
        generator,
        size,
        dim,
        path,
        num_queries,
        payload_gen,
        condition_gen,
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
        condition_generator=condition_gen,
    )
