import os
import json
import multiprocessing as mp
import random
from enum import IntEnum
from typing import List, Dict

import numpy as np
import tqdm

from generators.config import DATA_DIR
from generators.generate import DataGenerator


class ConditionType(IntEnum):
    date = 0
    category = 1
    submitter = 2


class ArxivGenerator:
    vectors: np.ndarray
    payloads: List[dict] = []
    filters: Dict[str, list] = {}
    generator: DataGenerator

    @classmethod
    def _init_generator(cls, vectors_path, payload_path, filters_path):
        print("init process")
        cls.vectors = cls._read_vectors(vectors_path)
        cls.payloads = cls._read_payload(payload_path)
        cls.filters = cls._read_filters(filters_path)
        cls.generator = DataGenerator()

    @classmethod
    def _read_vectors(cls, vectors_path):
        print("loading vectors")
        vectors = np.load(vectors_path, mmap_mode="r")
        print(f"vectors loaded, shape: {vectors.shape}, started loading payload")
        return vectors

    @classmethod
    def _read_payload(cls, payload_path):
        print("loading payload")
        with open(payload_path, "r") as fd:
            payload = [json.loads(line) for line in fd]
        print(f"payload loaded, len: {len(payload)}")
        return payload

    @classmethod
    def _read_filters(cls, filter_path):
        print("loading filters")
        with open(filter_path, "r") as fp:
            filters = json.load(fp)
        print("filters loaded, start generating queries")
        return filters

    @staticmethod
    def generate_condition(filters: dict):

        if (case := random.randint(0, 1)) == ConditionType.date:
            ts_range = filters["timestamp_range"]
            q25 = ts_range["q25"]
            q75 = ts_range["q75"]
            middle = (q25 + q75) // 2

            start = random.randint(q25, middle)
            end = random.randint(middle, q75)

            condition = {
                "and": [{"update_date_ts": {"range": {"gt": start, "lt": end}}}]
            }

        elif case == ConditionType.category:
            value = random.choice(filters["labels"])
            condition = {"and": [{"label": {"match": {"value": value}}}]}
        else:
            raise ValueError(f"Unrecognized option: <{case}>.")

        return condition

    @classmethod
    def search_one(cls, top):
        condition = dict()
        best_scores = list()
        closest_ids = list()

        query_vector = cls.vectors[random.randint(0, len(cls.vectors))]
        while len(closest_ids) < top:
            condition = cls.generate_condition(cls.filters)
            closest_ids, best_scores = cls.generator.search(
                vectors=cls.vectors,
                payloads=cls.payloads,
                query=query_vector,
                conditions=condition,
                top=top,
            )

        return json.dumps(
            {
                "query": query_vector.tolist(),
                "conditions": condition,
                "closest_ids": closest_ids,
                "closest_scores": best_scores,
            }
        )

    @classmethod
    def generate(
        cls,
        vectors_path,
        payload_path,
        filters_path,
        num_queries: int,
        top=10,
        parallel=1,
        output_path="out.jsonl",
    ):
        with open(output_path, "w") as f:
            if parallel == 1:
                cls._init_generator(vectors_path, payload_path, filters_path)
                for _ in tqdm.trange(num_queries):
                    f.write(cls.search_one(top) + "\n")
            else:
                with mp.Pool(
                    processes=parallel,
                    initializer=cls._init_generator,
                    initargs=(vectors_path, payload_path, filters_path),
                ) as pool:
                    with tqdm.tqdm(total=num_queries) as p_bar:
                        for json_result in pool.imap(
                            cls.search_one, (top for _ in range(num_queries))
                        ):
                            f.write(json_result + "\n")
                            p_bar.update(1)


if __name__ == "__main__":
    VECTORS_PATH = os.path.join(DATA_DIR, "arxiv", "vectors.npy")
    PAYLOAD_PATH = os.path.join(DATA_DIR, "arxiv", "payloads.jsonl")
    PREPARED_PAYLOAD_PATH = os.path.join(DATA_DIR, "arxiv", "prepared_payload.jsonl")
    FILTER_PATH = os.path.join(DATA_DIR, "arxiv", "filters.json")
    OUTPUT_PATH = os.path.join(DATA_DIR, "arxiv", "tests.jsonl")
    NUM_QUERIES = 100
    TOP = 10
    PARALLEL = 8

    if not os.path.exists(PREPARED_PAYLOAD_PATH):
        with open(PAYLOAD_PATH, "r") as src:
            with open(PREPARED_PAYLOAD_PATH, "w") as dest:
                for json_line in src:
                    line = json.loads(json_line)
                    line["label"] = random.choice(line["labels"])
                    dest.write(
                        json.dumps(
                            {
                                k: v
                                for k, v in line.items()
                                if k in {"update_date_ts", "label"}
                            }
                        )
                        + "\n"
                    )

    arxiv_generator = ArxivGenerator
    arxiv_generator.generate(
        VECTORS_PATH,
        PREPARED_PAYLOAD_PATH,
        FILTER_PATH,
        NUM_QUERIES,
        TOP,
        PARALLEL,
        OUTPUT_PATH,
    )
