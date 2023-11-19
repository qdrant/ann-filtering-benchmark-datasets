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
    path = DATA_DIR_PATH / "yandex_1B" / "t2i"
    path.mkdir(parents=True, exist_ok=True)

    I, D = knn_result_read(path / "t2i_new_groundtruth.public.100K.bin")
    V = read_fbin(path / "query.public.100K.fbin")

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
    # axel -n 100 --alternate https://storage.yandexcloud.net/yandex-research/ann-datasets/t2i_new_groundtruth.public.100K.bin
    # axel -n 100 --alternate https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.public.100K.fbin
    # axel -n 100 --alternate https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.1B.fbin
    main()
