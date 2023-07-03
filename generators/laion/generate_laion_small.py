import json
import os
import random
from typing import Iterable, Tuple

import pandas as pd
import numpy as np
import tqdm

from generators.config import DATA_DIR
from generators.search_generator.qdrant_generator import index_qdrant, search_qdrant

SAMPLE_SIZE = 100_000


def get_img_url(part: int) -> str:
    return f"https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings/img_emb/img_emb_{part}.npy"


def get_metadata_url(part: int) -> str:
    return f"https://deploy.laion.ai/8f83b608504d46bb81708ec86e912220/embeddings/metadata/metadata_{part}.parquet"


LAION_400M_embeddings_img_url = get_img_url(0)
LAION_400M_embeddings_metadata_url = get_metadata_url(0)


def download_file(url: str, path: str):
    print(f"Downloading {url} to {path}")
    os.system(f"wget {url} -O {path}")


def download():
    # make dirs
    os.makedirs(os.path.join(DATA_DIR, "laion"), exist_ok=True)
    # download files
    download_file(LAION_400M_embeddings_img_url, os.path.join(DATA_DIR, "laion", "img_emb.npy"))
    download_file(LAION_400M_embeddings_metadata_url, os.path.join(DATA_DIR, "laion", "metadata.parquet"))


def generate_queries(
        embeddings: np.ndarray,
        other_embeddings: np.ndarray,
        metadata: pd.DataFrame,
        n: int = 10
) -> Iterable[dict]:
    other_embeddings_len = other_embeddings.shape[0]

    for _ in tqdm.tqdm(range(n)):
        # get random embedding
        random_idx = np.random.randint(other_embeddings_len)
        random_embedding = other_embeddings[random_idx]

        # get distance
        distances = np.linalg.norm(embeddings - random_embedding, axis=1)

        # get closest
        closest_idx = np.argmin(distances)
        closest_embedding = embeddings[closest_idx]

        # get metadata
        closest_metadata = metadata.iloc[closest_idx]

        yield {
            "query": random_embedding.tolist(),
            "closest": closest_embedding.tolist(),
            "metadata": closest_metadata.to_dict()
        }


def filter_generator() -> Tuple[dict, dict]:
    if random.random() < 0.5:

        # Score range from 0.3 to 0.4
        min_score = random.uniform(0.3, 0.4)

        return (
            {
                "and": [
                    {
                        "similarity": {
                            "range": {
                                "gt": min_score,
                            }
                        }
                    }
                ]
            },
            {
                "must": [
                    {
                        "key": "similarity",
                        "range": {
                            "gt": min_score,
                        }
                    }
                ]
            }
        )
    else:
        return {}, {}


def main():
    # download()

    # read parquet
    df = pd.read_parquet(os.path.join(DATA_DIR, "laion", "metadata.parquet"))

    # get embeddings
    embeddings = np.load(os.path.join(DATA_DIR, "laion", "img_emb.npy"))

    df.fillna(0, inplace=True)

    # First SAMPLE_SIZE rows
    df_sample = df.iloc[:SAMPLE_SIZE]
    embeddings_sample = embeddings[:SAMPLE_SIZE]
    other_embeddings = embeddings[SAMPLE_SIZE:]

    payload = df_sample.to_dict(orient="records")

    index_qdrant(embeddings_sample, payload)

    path = os.path.join(DATA_DIR, "laion", "small")

    tests_path = os.path.join(path, "tests.jsonl")

    with open(tests_path, "w") as f:
        for query in tqdm.tqdm(search_qdrant(
                sample_embeddings=other_embeddings,
                filter_generator=filter_generator,
                n=5000,
                top=10
        )):
            f.write(f"{json.dumps(query)}\n")

    # save embeddings

    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, "vectors.npy"), embeddings_sample)

    # save metadata
    payload_path = os.path.join(path, "payloads.jsonl")
    df_sample.to_json(payload_path, orient="records", lines=True)


if __name__ == '__main__':
    main()
