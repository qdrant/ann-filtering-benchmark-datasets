# ANN Filtered Retrieval Datasets

This repo contains a collection of datasets, inspired by [ann-benchmarks](https://github.com/erikbern/ann-benchmarks) for searching for similar vectors with additional filtering conditions.

## Motivation

More and more applications are now using vector similarity search in their products.
The task of approximate nearest neighbor (ANN) search has gone beyond the scope of academic research and the narrow circle of huge IT corporations. 

In this regard, the issue of supplementing vector search with application business logic is becoming more and more relevant.

## Examples and cases

It is no longer enough to simply search for similar dishes by photo, you only need to search for them in those restaurants that are in the delivery area.

It is not enough to search for all items similar by description, you also need to consider price ranges, stock availability, etc.

It's not enough to find candidates for a job position based on similar skills, you also have to consider location, level of spoken language, and seniority.

You name it.

## Is it that different?

Classical approaches to ANN, and their implementations in many libraries, were usually customized for benchmarks, where the search speed among all vectors is the only comparison criterion.

Because of this, they had to sacrifice many functions that are useful in other situations: the ability to quickly delete, insert and modify stored values, as well as saving and  filtering based on metadata.

## Data

| description                      | Num vectors | dim  | distance | filters               | link                                                                                            |
|----------------------------------|-------------|------|----------|-----------------------|-------------------------------------------------------------------------------------------------|
| all-MiniLM-L6-v2 ArXiv titles    | 2 138 591   | 384  | Cosine   | match keyword / range | [link](https://storage.googleapis.com/ann-filtered-benchmark/datasets/arxiv.tar.gz)             | 
| Efficientnet encoded H&M Clothes | 105 100     | 2048 | Cosine   | match keyword         | [link](https://storage.googleapis.com/ann-filtered-benchmark/datasets/hnm.tgz)                  |
| Random vectors \ random payload  | 1 000 000   | 100  | Cosine   | match keyword         | [link](https://storage.googleapis.com/ann-filtered-benchmark/datasets/random_keywords_1m.tgz)   |
| Random vectors \ random payload  | 1 000 000   | 100  | Cosine   | match int             | [link](https://storage.googleapis.com/ann-filtered-benchmark/datasets/random_ints_1m.tgz)       |
| Random vectors \ random payload  | 1 000 000   | 100  | Cosine   | range                 | [link](https://storage.googleapis.com/ann-filtered-benchmark/datasets/random_float_1m.tgz)      |
| Random vectors \ random payload  | 1 000 000   | 100  | Cosine   | geo-radius            | [link](https://storage.googleapis.com/ann-filtered-benchmark/datasets/random_geo_1m.tgz)        |
| Random vectors \ random payload  | 100 000     | 2048 | Cosine   | match keyword         | [link](https://storage.googleapis.com/ann-filtered-benchmark/datasets/random_keywords_100k.tgz) |
| Random vectors \ random payload  | 100 000     | 2048 | Cosine   | match int             | [link](https://storage.googleapis.com/ann-filtered-benchmark/datasets/random_ints_100k.tgz)     |
| Random vectors \ random payload  | 100 000     | 2048 | Cosine   | range                 | [link](https://storage.googleapis.com/ann-filtered-benchmark/datasets/random_float_100k.tgz)    |
| Random vectors \ random payload  | 100 000     | 2048 | Cosine   | geo-radius            | [link](https://storage.googleapis.com/ann-filtered-benchmark/datasets/random_geo_100k.tgz)      |

### Data Format

Each dataset contains of following files:

* `vectors.npy` - Numpy matrix of vectors. Shape `num_vectors x dim`
* `payloads.jsonl` - payload values, associated with vectors. Number of lines equal to `num_vectors`
* `tests.jsonl` - collection of queries with filtering conditions and expected results. Contains fields:
  * `query` - vector to be used for similarity search
  * `conditions` - filtering conditions of 3 possible types: `match`, `range`, and `geo`
  * `closest_ids` - IDs of records, expected to be found with given query
  * `closest_scores` - similarity scores of associated IDs

### Sources

* Random data generator - [script](./generators/random_data)
* Image data - [kaggle](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
* Image embeddings generator - [colab](https://colab.research.google.com/drive/1u5-gZjPzfDP50c7LQztlVd78kGPyTAb1?usp=sharing)
