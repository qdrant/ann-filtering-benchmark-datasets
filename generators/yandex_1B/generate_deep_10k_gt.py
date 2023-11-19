import json
import os
import tqdm
import sys
sys.path.append(".")

from pathlib import Path

from generators.yandex_1B import read_fbin, knn_result_read
from generators.config import DATA_DIR


def main():
    print("Loading Yandex Ground Truth dataset")

    DATA_DIR_PATH = Path(DATA_DIR)
    path = DATA_DIR_PATH / "yandex_1B" / "deep"
    path.mkdir(parents=True, exist_ok=True)

    I, D = knn_result_read(path / "deep_new_groundtruth.public.10K.bin")
    V = read_fbin(path / "query.public.10K.fbin")

    tests_path = os.path.join(path, "tests.jsonl")

    with open(tests_path, "w") as f:
        for vector, expected_result, expected_scores in tqdm.tqdm(zip(V, I, D), total=len(V)):
            query = {
                "query": vector.tolist(),
                "conditions": {},
                "closest_ids": [hit_id for hit_id in expected_result.tolist()],
                "closest_scores": [score for score in expected_scores.tolist()]
            }
            f.write(f"{json.dumps(query)}\n")

if __name__ == '__main__':
    # download:
    # axel -n 100 --alternate https://storage.yandexcloud.net/yandex-research/ann-datasets/deep_new_groundtruth.public.10K.bin
    # axel -n 100 --alternate https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/query.public.10K.fbin
    # axel -n 100 --alternate https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin
    main()
