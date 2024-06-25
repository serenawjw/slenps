from abc import ABC, abstractmethod

from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, Doc2Vec
from sentence_transformers import SentenceTransformer
import logging

import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
from typing import Union, List, Any

from slenps.utils import check_memory_usage

# Initialize logging
logging.basicConfig(level=logging.INFO)

class BaseEM(ABC):
    """All embedding model should implement this base class"""
    def __init__(self):
        self.model = None

    @abstractmethod
    def encode(self, documents):
        """Method to encode documents, to be implemented by all subclasses."""
        pass

class TfidfEM(BaseEM):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = TfidfVectorizer(**kwargs)
        logging.info("TfidfEM initialized with specified parameters.")

    def encode(self, documents):
        result = self.model.fit_transform(documents)
        logging.info(f"Output dimensions: {result.shape[1]}")
        return result.toarray()

class Word2VecEM(BaseEM):
    def __init__(self, size=100, **kwargs):
        super().__init__()
        self.size = size
        self.model = Word2Vec(vector_size=size, min_count=1, **kwargs)
        logging.info(f"Word2Vec model initialized with {size} dimensions")

    def encode(self, documents):
        self.model.build_vocab(documents)
        self.model.train(documents, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        logging.info(f'Output dimensions: {self.size}')
        return np.array(
            [np.mean([self.model.wv[word] for word in doc if word in self.model.wv], axis=0) for doc in documents]
        )

class Doc2VecEM(BaseEM):
    def __init__(self, size=100):
        super().__init__()
        self.size = size
        self.model = Doc2Vec(vector_size=size, min_count=1, epochs=10)
        logging.info(f"Doc2Vec model initialized with {size} dimensions")

    def encode(self, documents):
        self.model.build_vocab(documents)
        self.model.train(documents, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        return np.array([self.model.infer_vector(doc.words) for doc in documents])

class SbertEM(BaseEM):
    def __init__(self, model_name, **kwargs):
        super().__init__()
        self.model = SentenceTransformer(model_name, **kwargs)
        logging.info(f'Sbert model initialized with {self.model.get_sentence_embedding_dimension()} dimensions')

    def encode(self, documents):
        embeddings = self.model.encode(documents)
        logging.info(f'Output dimensions: {len(embeddings[0])}')
        return embeddings

def load_embedding_model(model_name: str = 'all-MiniLM-L6-v2', mode: str = None, **kwargs):
    """
    Load a model based on the mode and model name.

    Args:
    model_name (str): Name of the model to load.
    mode (str): Type of model to load ('huggingface', 'tfidf', 'word2vec', 'doc2vec').

    Returns:
    Model with an .encode method or equivalent functionality.
    """
    if mode == 'huggingface':
        return SbertEM(model_name, **kwargs)
    elif model_name == 'tfidf':
        return TfidfEM(**kwargs)
    elif model_name == 'word2vec':
        return Word2VecEM(**kwargs)
    elif model_name == 'doc2vec':
        return Doc2VecEM(**kwargs)
    else:
        raise ValueError("Unsupported mode specified. Choose from 'huggingface' by specifying model name and mode as 'huggingface', or set model_name as 'tfidf', 'word2vec' or 'doc2vec'.")


def embed_documents(model: BaseEM, documents: Union[np.ndarray, pd.Series, list], pickle_filepath: Union[Path, str]):
    """
    Embeds documents using the embedding model and saves the embeddings along with the documents into a pickle file.

    Args:
    model (BaseEM): The embedding model to use.
    documents (Union[np.ndarray, pd.Series, list]): The documents to be embedded.
    pickle_filepath (Union[Path, str]): Path where the pickle file will be stored.
    
    Raises:
    ValueError: If the output file already exists to prevent overwriting of data.
    """
    if not isinstance(documents, np.ndarray):
        documents = np.array(documents)
    documents = np.unique(documents)

    embeddings = model.encode(documents)

    output_path = os.path.join(os.path.dirname(pickle_filepath), 'embedding.pickle')
    if os.path.exists(output_path):
        raise ValueError(f"File already exists at {output_path}")

    with open(output_path, 'wb') as file:
        pickle.dump((embeddings, documents), file)
        logging.info(f"embeddings and documents saved to {output_path}")


def get_data_from_paths(filepaths: List[Union[Path, str]]):
    embeddings_list = []
    documents_list = []

    for filepath in filepaths:

        if not os.path.exists(filepath):
            raise ValueError(f'No file named {filepath}!')

        with open(filepath, 'rb') as file:
            obj = pickle.load(file)

        embeddings_list.append(obj[0])
        documents_list.append(obj[1])

    concatenated_embeddings = np.vstack(embeddings_list) if embeddings else np.array([])
    concatenated_documents = np.concatenate(documents_list) if documents else np.array([])

    check_memory_usage(concatenated_documents, concatenated_embeddings)
    return np.concatenate(embeddings), np.concatenate(documents)



# Example usage:
# Assuming 'model' is an instance of BaseEM or any derived class (loaded using load_model function)
# documents = ['This is a document.', 'Here is another one.']
# embed_documents(model, documents, '/path/to/save/embedding.pickle')

