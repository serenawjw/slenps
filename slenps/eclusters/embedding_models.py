"""
This module defines a registry for embedding models and provides implementations
for various embedding techniques including TF-IDF, Word2Vec, Doc2Vec, and SBERT.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Type, Optional, Any, Union

import numpy as np
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, Doc2Vec
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingModelRegistry(type):
    """
    A metaclass for registering embedding model classes.
    """

    REGISTRY: Dict[str, Type["BaseEmbeddingModel"]] = {}

    def __new__(
        cls, name: str, bases: tuple, attrs: dict
    ) -> Type["BaseEmbeddingModel"]:
        new_cls = super().__new__(cls, name, bases, attrs)
        cls.REGISTRY[name] = new_cls
        return new_cls

    @classmethod
    def get_registry(cls, name: str) -> Optional[Type["BaseEmbeddingModel"]]:
        """
        Retrieve an embedding model from the registry by name.

        Args:
            name: The name of the embedding model.

        Returns:
            The requested embedding model class, or None if not found.
        """
        return cls.REGISTRY.get(name)


class BaseEmbeddingModel(ABC, metaclass=EmbeddingModelRegistry):
    """
    Abstract base class for all embedding models.
    """

    model: Any

    @abstractmethod
    def encode(self, documents: List[str]) -> np.ndarray:
        """
        Method to encode documents, to be implemented by all subclasses.

        Args:
            documents: List of documents to encode.

        Returns:
            Numpy array of document embeddings.
        """
        pass


class TfidfEM(BaseEmbeddingModel):
    """
    TF-IDF embedding model.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.model = TfidfVectorizer(**kwargs)
        logger.info("TfidfEM initialized with specified parameters.")

    def encode(self, documents: List[str]) -> np.ndarray:
        result = self.model.fit_transform(documents)
        logger.info(f"Output dimensions: {result.shape[1]}")
        return result.toarray()


class Word2VecEM(BaseEmbeddingModel):
    """
    Word2Vec embedding model.
    """

    def __init__(self, size: int = 100, **kwargs: Any) -> None:
        self.size = size
        self.model = Word2Vec(vector_size=size, min_count=1, **kwargs)
        logger.info(f"Word2Vec model initialized with {size} dimensions")

    def encode(self, documents: List[List[str]]) -> np.ndarray:
        self.model.build_vocab(documents)
        self.model.train(
            documents, total_examples=self.model.corpus_count, epochs=self.model.epochs
        )
        logger.info(f"Output dimensions: {self.size}")
        return np.array(
            [
                np.mean(
                    [self.model.wv[word] for word in doc if word in self.model.wv],
                    axis=0,
                )
                for doc in documents
            ]
        )


class Doc2VecEM(BaseEmbeddingModel):
    """
    Doc2Vec embedding model.
    """

    def __init__(self, size: int = 100, **kwargs: Any) -> None:
        self.size = size
        self.model = Doc2Vec(vector_size=size, min_count=1, epochs=10, **kwargs)
        logger.info(f"Doc2Vec model initialized with {size} dimensions")

    def encode(self, documents: List[Any]) -> np.ndarray:
        self.model.build_vocab(documents)
        self.model.train(
            documents, total_examples=self.model.corpus_count, epochs=self.model.epochs
        )
        return np.array([self.model.infer_vector(doc.words) for doc in documents])


class SbertEM(BaseEmbeddingModel):
    """
    Sentence-BERT embedding model.
    """

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self.model = SentenceTransformer(model_name, **kwargs)
        logger.info(
            f"SBERT model initialized with {self.model.get_sentence_embedding_dimension()} dimensions"
        )

    def encode(self, documents: List[str]) -> np.ndarray:
        embeddings = self.model.encode(documents)
        logger.info(f"Output dimensions: {len(embeddings[0])}")
        return embeddings
