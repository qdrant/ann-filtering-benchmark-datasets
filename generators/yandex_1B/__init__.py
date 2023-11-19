import numpy as np
import os

def read_fbin(filename, start_idx=0, chunk_size=None, memmap=False):
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

def read_single_vector(filename, index):
    """ Read a single float32 vector from a *.fbin file at a given index
    Args:
        :param filename (str): path to *.fbin file
        :param index (int): index of the vector to read
    Returns:
        A single float32 vector (numpy.ndarray)
    """
    with open(filename, "rb") as f:
        total_vectors, dim = np.fromfile(f, count=2, dtype=np.int32)

        # Check if the index is within the range of available vectors
        if index < 0 or index >= total_vectors:
            raise ValueError(f"Index {index} is out of bounds for a file with {total_vectors} vectors.")

        # Calculate the offset in bytes to reach the desired vector
        # offset = 2 * 4 + index * dim * 4  # 2 integers for nvecs and dim, each 4 bytes
        offset = np.int64(2 * 4) + np.int64(index) * np.int64(dim) * np.int64(4)

        # Read the single vector
        f.seek(offset)
        vector = np.fromfile(f, count=dim, dtype=np.float32)

    return vector

def knn_result_read(fname):
    n, d = map(int, np.fromfile(fname, dtype="uint32", count=2))
    assert os.stat(fname).st_size == 8 + n * d * (4 + 4)
    f = open(fname, "rb")
    f.seek(4+4)
    I = np.fromfile(f, dtype="int32", count=n * d).reshape(n, d)
    D = np.fromfile(f, dtype="float32", count=n * d).reshape(n, d)
    return I, D
