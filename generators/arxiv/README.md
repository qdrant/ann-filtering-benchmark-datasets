## Arxiv queries generator

Steps to generate arxiv queries:
1. Download data from [kaggle arxiv dataset](https://www.kaggle.com/Cornell-University/arxiv)
2. Preprocess the data
3. Calculate embeddings
4. Create filters
5. Generate arxiv queries  

First four steps can be found in the following [colab notebook](https://colab.research.google.com/drive/1aYGQ9JLKclc7CIxJKIwWos9KaFcUFv2s).

Currently, supported 3 filters based on:
- Update date timestamp
- Submitter
- Category

---
**Caution:**

Kaggle arxiv dataset contains ~2.2 millions records. 

It may require huge amount of memory to run it.

We perform a brute-force approach to find the nearest neighbors of a point, therefore the process of generating arxiv queries is time-consuming. 

---


