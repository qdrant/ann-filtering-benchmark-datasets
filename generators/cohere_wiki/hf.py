# This file contains function to work with Hugging Face datasets on a lower level.
# Instead of using `load_dataset` function, which is not optimized and loads all data to local machine (streaming doesn't really work),
# we will download files directly and read them one by one.

import os
from typing import Iterable, Optional
from huggingface_hub import HfApi, hf_hub_download
import pandas as pd
import time
import shutil
from multiprocessing import Process, Queue
HF_CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


def list_files(dataset_name: str, split: str = "train") -> list[str]:
    """
    List all files in the dataset.
    
    Args:
        dataset_name: Name of the Hugging Face dataset
        split: Dataset split (train, validation, test)
        
    Returns:
        List of file names in the dataset
    """
    api = HfApi()
    try:
        files = api.list_repo_files(dataset_name, repo_type="dataset")
        files = [f for f in files if split in f]
        # Sort files by name, to make it deterministic
        files = sorted(files)
        return files
    except Exception as e:
        print(f"Error listing files: {e}")
        return []

def _download_worker(dataset_name: str, split: str, file_name: str, queue: Queue):
    """
    Worker function that downloads a file and puts the result in a queue.
    """
    try:
        local_path = hf_hub_download(
            repo_id=dataset_name,
            filename=file_name,
            repo_type="dataset",
            cache_dir=HF_CACHE_DIR
        )
        queue.put(local_path)
    except Exception as e:
        print(f"Error downloading file: {e}")
        queue.put(None)

def download_file_async(dataset_name: str, split: str = "train", file_name: str = "data.jsonl") -> Queue:
    """
    Download a file from the dataset asynchronously.
    This function creates a parallel process to download the file.
    
    Args:
        dataset_name: Name of the Hugging Face dataset
        split: Dataset split (train, validation, test)
        file_name: Name of the file to download
        
    Returns:
        Path to the downloaded file (once download is complete)
    """
    # Create a queue to communicate with the worker process
    queue = Queue()
    
    # Start the download process
    process = Process(
        target=_download_worker,
        args=(dataset_name, split, file_name, queue)
    )
    process.start()
    
    # Return the queue - the caller can use queue.get() to get the result
    return queue


def download_file(dataset_name: str, split: str = "train", file_name: str = "data.jsonl") -> Optional[str]:
    """
    Download a file from the dataset.
    
    Args:
        dataset_name: Name of the Hugging Face dataset
        split: Dataset split (train, validation, test)
        file_name: Name of the file to download
        
    Returns:
        Path to the downloaded file
    """
    try:
        # Download file using huggingface_hub
        local_path = hf_hub_download(
            repo_id=dataset_name,
            filename=file_name,
            repo_type="dataset",
            cache_dir=HF_CACHE_DIR
        )
        return local_path
    except Exception as e:
        print(f"Error downloading file: {e}")
        return None

def clear_hf_cache():
    """
    Clear the local cache for a Hugging Face dataset.
    """
    if os.path.exists(HF_CACHE_DIR):
        shutil.rmtree(HF_CACHE_DIR)


def read_dataset_stream(dataset_name: str, split: str = "train") -> Iterable[dict]:
    """
    Read the dataset as a stream.

    - List all files in the dataset
    - Find parquet files
    - One by one:
        - Download the file
        - Read the file as a stream
        - Yield each object in the file
        - Delete the file after yielding

    Args:
        dataset_name: Name of the Hugging Face dataset
        split: Dataset split (train, validation, test)
        
    Yields:
        Dictionary containing the data for each row
    """
    # List all files in the dataset
    files = list_files(dataset_name, split)
    print(f"Found files: {files}")
    
    # Filter for parquet files
    parquet_files = [f for f in files if f.endswith('.parquet')]
    print(f"Found parquet files: {parquet_files}")

    for i, file_name in enumerate(parquet_files):
        # Download the file
        print(f"Downloading file {i}...")
        local_path = download_file(dataset_name, split, file_name)

        # Run a parallel process to download one file ahead
        if i < len(parquet_files) - 1:
            print(f"Async Downloading file {i + 1}...")
            download_file_async(dataset_name, split, parquet_files[i + 1])

        if not local_path:
            continue
            
        try:
            # Read parquet file
            df = pd.read_parquet(local_path)
            for row in df.itertuples():
                yield row._asdict()
        finally:
            # Clean up the downloaded file
            if os.path.exists(local_path):
                os.remove(local_path)

        if i % 10 == 0:
            # Prevent accumulating cache on disk
            clear_hf_cache()


def main():
    # Using a public dataset as an example
    dataset_name = "Cohere/wikipedia-22-12-simple-embeddings"
    print(f"Listing files for dataset: {dataset_name}")

    total = 0
    # Read 10000 rows and measure time
    start_time = time.time()
    for i, item in enumerate(read_dataset_stream(dataset_name, "train")):
        if i >= 100000:
            break
        total += len(item["emb"].tolist())
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Total: {total}")

if __name__ == "__main__":
    main()




