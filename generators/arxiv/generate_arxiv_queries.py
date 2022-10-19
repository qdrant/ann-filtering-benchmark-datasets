import json
import os
import random
from enum import IntEnum
from typing import List, Dict

import numpy as np
from tqdm import tqdm

from generators.config import DATA_DIR
from generators.generate import DataGenerator


class ConditionType(IntEnum):
    date = 0
    category = 1
    submitter = 2


def generate_query(filters: dict):

    if (case := random.randint(0, 2)) == ConditionType.date:
        ts_range = filters["timestamp_range"]
        q25 = ts_range["q25"]
        q75 = ts_range["q75"]
        middle = (q25 + q75) // 2

        start = random.randint(q25, middle)
        end = random.randint(middle, q75)

        condition = {"and": [{"update_date_ts": {"range": {"gt": start, "lt": end}}}]}
    elif case == ConditionType.category:
        value = random.choice(filters["labels"])
        condition = {"and": [{"labels": {"match": {"value": value}}}]}
    elif case == ConditionType.submitter:
        value = random.choice(filters["top_submitters"])
        condition = {"and": [{"submitter": {"match": {"value": value}}}]}
    else:
        raise ValueError(f"Unrecognized option: <{case}>.")

    return condition


def generate_arxiv_queries(
    vectors: np.ndarray,
    payloads: List[dict],
    filters: Dict[str, list],
    num_queries: int,
    path,
    top=25
):
    generator = DataGenerator()
    repeat_count = 0
    with open(path, "w") as out:
        for _ in tqdm(range(num_queries)):
            closest_ids = []
            local_repeat = 0
            while len(closest_ids) < top:
                ref_id = random.randint(0, len(vectors))
                query_vector = vectors[ref_id]
                query_filter = generate_query(filters=filters)

                # precise search
                closest_ids, best_scores = generator.search(
                    vectors=vectors,
                    payloads=payloads,
                    query=query_vector,
                    conditions=query_filter,
                    top=top
                )
                if len(closest_ids) < top:
                    repeat_count += 1
                    local_repeat += 1
                    print(f'found only {len(closest_ids)} responses. '
                          f'Repeat ({repeat_count}, {repeat_count})')

            out.write(
                json.dumps(
                    {
                        "query": query_vector.tolist(),
                        "conditions": query_filter,
                        "closest_ids": closest_ids,
                        "closest_scores": best_scores,
                    }
                )
            )

            out.write("\n")


if __name__ == "__main__":
    print('loading vectors')
    vectors_path = os.path.join(
        DATA_DIR, "arxiv", "vectors.npy"
    )
    vectors = np.load(vectors_path, mmap_mode="r")
    print(f'vectors loaded, shape: {vectors.shape}, started loading payload')
    payloads_path = os.path.join(DATA_DIR, "arxiv", "payloads.jsonl")
    payloads = []
    with open(payloads_path, "r") as fd:
        for line in tqdm(fd):
            payloads.append(json.loads(line))
    print(f'payloads loaded, len: {len(payloads)}')
    filters_path = os.path.join(DATA_DIR, "arxiv", "filters.json")
    print('loading filters')
    with open(filters_path, "r") as fp:
        filters = json.load(fp)
    print('filters loaded, start generating queries')

    generate_arxiv_queries(
        vectors=vectors,
        payloads=payloads,
        filters=filters,
        num_queries=10000,
        path=os.path.join(DATA_DIR, "arxiv", "tests.jsonl"),
    )
