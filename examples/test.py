from slenps.eclusters import load_embedding_model, get_clustering_model_dict, load_clustering_model, cluster, find_best_algorithm
import numpy as np 
import pandas as pd 

# load documents

with open('sample_documents.txt', 'r') as file:
    documents = np.array([line.strip() for line in file.readlines()])

# embedding model
# embedding_model = load_embedding_model(
#     model_name = 'all-MiniLM-L6-v2', mode = 'huggingface',
# )
embedding_model = load_embedding_model(model_name='word2vec') 

# obtain embeddings 
embeddings = embedding_model.encode(documents)

# clustering model
clustering_model = load_clustering_model('kmeans')
clustering_model = clustering_model.set_params(n_clusters=3)

# fit the model and retrieve labels and metrics
labels, metrics = cluster(
    embeddings, clustering_model, 
    metrics = ['dbs', 'calinski'],
)
print(metrics)

# print sample result
n_samples = 10
for document, label in zip(documents[:n_samples], labels[:n_samples]):
    print(f'{document} -> label {label}')


# define a list of clustering models to evaluate
# all default models are included in get_clustering_model_dict
model_names = ['kmeans', 'agglomerative_clustering', 'spectral_clustering']

# find best algo and num_cluster using test_metric
results = find_best_algorithm(
    embeddings, model_names=model_names,
    test_metric='dbs', metrics = ['dbs', 'silhouette'],
    min_cluster_num=2, max_cluster_num=10,
    result_filepath='sample_result_metrics.csv',
    print_topk=True,
)

# view all results
print(pd.DataFrame(results))