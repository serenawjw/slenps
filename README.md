# slenps

slenps is a collection of some simple NLP pipelines
- ecluster: cluster word embeddings
- llmclf: binary and multiclass classification using LLM

## Installation
### PyPI 
```bash
pip install slenps
```


## Example usage

### embed documents and cluster its embeddings

Cluster some word with chosen embedding and clustering models

```py
from slenps.eclusters import load_embedding_model, get_clustering_model_dict, load_clustering_model, cluster
import numpy as np 

## Obtain documents and embeddings
with open('sample_documents.txt', 'r') as file:
    documents = np.array([line.strip() for line in file.readlines()])

# get embedding model
# embedding_model = load_embedding_model(
#     model_name='all-MiniLM-L6-v2', mode='huggingface'
# )
embedding_model = load_embedding_model(model_name="Word2VecEM")

# embed documents
embeddings = embedding_model.encode(documents)
print(f"Embedding shape: {embeddings.shape}\nDocuments shape: {documents.shape}")


## Clustering

# Select a clustering model and number of clusters
model_name = "kmeans"
num_cluster = 3

# create a clustering model
clustering_model = load_clustering_model(model_name).set_params(n_clusters=num_cluster)
clustering_model

# fit the model and retrieve labels and metrics
labels, metrics = cluster(
    embeddings,
    clustering_model,
    metrics=["dbs", "silhouette", "calinski"],
    return_model=False,
)
print(f"Clustering metrics: {metrics}")

# print sample result
n_samples = 10
for document, label in zip(documents[:n_samples], labels[:n_samples]):
    print(f"{document} --> Label {label}")
```

## Find the best algorithm and num_cluster 
```py
from slenps.eclusters import find_best_algorithm
import pandas as pd
# define a list of clustering models to evaluate
# all default models are included in get_clustering_model_dict
model_names = ['kmeans', 'agglomerative_clustering', 'spectral_clustering']

# find best algo and num_cluster using test_metric
results = find_best_algorithm(
    embeddings,
    model_names=model_names,
    metrics=["dbs", "silhouette"],
    test_metric="dbs",
    min_cluster_num=2,
    max_cluster_num=10,
    result_filepath="sample_result_metric.csv",
    print_topk=True,
)

# view all results
pd.DataFrame(results)
```


## Supported models
### Embedding models

| embedding model | model_name | mode |
| :- | :-: | :-: |
| sklearn.feature.extraction.text.TfidfVectorizer | TfidfEM | None | 
| gensim.models.Word2Vec | Word2VecEM | None | 
| gensim.models.Doc2Vec | Doc2VecEM | None |
| sentence_transformers.SentenceTransformer | Any | huggingface | 

### Clustering models
| clustering model | model_name | default params |
| :- | :-: | :-: |
| sklearn.cluster.KMeans | kmeans | n_init='auto' | 
| sklearn.cluster.AgglomerativeClustering | agglomerative_clustering | None | 
| sklearn.cluster.SpectralClustering | spectral_clustering | None | 
| sklearn.cluster.MeanShift | mean_shift | None | 
| sklearn.cluster.AffinityPropagation | affinity_propagation | None | 
| sklearn.cluster.Birch | birch | threshold=0.2 | 



