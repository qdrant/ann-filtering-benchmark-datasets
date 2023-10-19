import json
import os
import random
from typing import Iterable, Tuple

import pandas as pd
import numpy as np
import tqdm

from pathlib import Path
from datasets import load_dataset
from generators.config import DATA_DIR
from generators.search_generator.qdrant_generator import index_qdrant, search_qdrant

SAMPLE_SIZE = 100_000


def main():
    print("Loading DBpedia OpenAI embeddings dataset")
    data = load_dataset("KShivendu/dbpedia-entities-openai-1M", split="train")

    embeddings = data.to_pandas()['openai'].to_numpy()
    embeddings = np.vstack(embeddings).reshape((-1, 1536))

    embeddings_sample = embeddings[:SAMPLE_SIZE]
    other_embeddings = embeddings[SAMPLE_SIZE:]

    print("Emb shape", other_embeddings.shape)

    index_qdrant(embeddings_sample, [])

    path = os.path.join(DATA_DIR, "dbpedia_openai", "1M")
    Path(path).mkdir(parents=True, exist_ok=True)

    tests_path = os.path.join(path, "tests.jsonl")

    with open(tests_path, "w") as f:
        for query in tqdm.tqdm(search_qdrant(
                sample_embeddings=other_embeddings,
                filter_generator=lambda: ({}, {}),
                n=5000,
                top=10
        )):
            f.write(f"{json.dumps(query)}\n")

    # save embeddings

    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, "vectors.npy"), embeddings_sample)


if __name__ == '__main__':
    main()
