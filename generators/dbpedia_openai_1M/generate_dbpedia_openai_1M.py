import json
import os

import numpy as np
import tqdm

from pathlib import Path
from datasets import load_dataset
from generators.config import DATA_DIR
from generators.search_generator.qdrant_generator import index_qdrant, search_qdrant

SAMPLE_SIZE = 900_000 # The dataset has 1 million embeddings in total


def main():
    print("Loading DBpedia OpenAI embeddings dataset")
    data = load_dataset("KShivendu/dbpedia-entities-openai-1M", split="train")

    embeddings = data.to_pandas()['openai'].to_numpy()
    embeddings = np.vstack(embeddings).reshape((-1, 1536))

    index_embeddings = embeddings[:SAMPLE_SIZE]
    query_embeddings = embeddings[SAMPLE_SIZE:]

    print("Shape of embeddings to be indexed", index_embeddings.shape)
    print("Shape of embeddings to be queried", query_embeddings.shape)

    index_qdrant(index_embeddings, [])

    path = os.path.join(DATA_DIR, "dbpedia_openai", "1M")
    Path(path).mkdir(parents=True, exist_ok=True)

    tests_path = os.path.join(path, "tests.jsonl")

    with open(tests_path, "w") as f:
        for query in tqdm.tqdm(search_qdrant(
                sample_embeddings=query_embeddings,
                filter_generator=lambda: ({}, {}),
                n=5000,
                top=10
        )):
            f.write(f"{json.dumps(query)}\n")

    # save embeddings

    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, "vectors.npy"), index_embeddings)


if __name__ == '__main__':
    main()
