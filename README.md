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

| description                     | Num vectors | dim  | distance | filters              | link     |
|---------------------------------|-------------|------|----------|----------------------|----------|
| Random vectors \ random payload | 1_000_000   | 100  | Cosine   | match                | [link]() |
| Random vectors \ random payload | 1_000_000   | 100  | Cosine   | range                | [link]() |
| Random vectors \ random payload | 1_000_000   | 100  | Cosine   | geo-radius           | [link]() |
| Random vectors \ random payload | 1_000_000   | 100  | Cosine   | multiple/combination | [link]() |
| Random vectors \ random payload | 100_000     | 2048 | Cosine   | match                | [link]() |
| Random vectors \ random payload | 100_000     | 2048 | Cosine   | range                | [link]() |
| Random vectors \ random payload | 100_000     | 2048 | Cosine   | geo-radius           | [link]() |
| Random vectors \ random payload | 100_000     | 2048 | Cosine   | multiple/combination | [link]() |


### Sources


* Random data generator - [ToDo]
* Image data https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations
* Image embeddings generator
