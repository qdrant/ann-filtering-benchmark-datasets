import json
import os

import numpy as np
import tqdm

from pathlib import Path
from generators.config import DATA_DIR

def read_fbin(filename, start_idx=0, chunk_size=None):
    """ Read *.fbin file that contains float32 vectors
    Args:
        :param filename (str): path to *.fbin file
        :param start_idx (int): start reading vectors from this index
        :param chunk_size (int): number of vectors to read.
                                 If None, read all vectors
    Returns:
        Array of float32 vectors (numpy.ndarray)
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.float32,
                          offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim)

def knn_result_read(fname):
    n, d = map(int, np.fromfile(fname, dtype="uint32", count=2))
    assert os.stat(fname).st_size == 8 + n * d * (4 + 4)
    f = open(fname, "rb")
    f.seek(4+4)
    I = np.fromfile(f, dtype="int32", count=n * d).reshape(n, d)
    D = np.fromfile(f, dtype="float32", count=n * d).reshape(n, d)
    return I, D


def main():
    print("Loading Yandex 100K Ground Truth dataset")

    DATA_DIR_PATH = Path(DATA_DIR)
    path = DATA_DIR_PATH / "yandex_t2i_1B" / "100K"
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
    main()
