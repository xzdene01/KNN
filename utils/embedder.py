# Part of this file if copied from Ing. Martin Fajčík Ph.D. and slightly modified

import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer


class StubEncoder:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def encode(self,
               docs: List[str],
               show_progress_bar: bool = False,
               normalize_embeddings: bool = False
               ):
        assert len(docs) == self.embeddings.shape[0]
        return self.embeddings

# class BasicSentenceEmbedder:
#     def __init__(
#         self,
#         model: str="all-MiniLM-L6-v2",
#         device: str="cpu",
#         normalize_embeddings: bool = False,
#         verbose: bool = False,
#     ):
#         self.verbose = verbose
#         self.normalize_embeddings = normalize_embeddings

#         if isinstance(model, str):
#             self.model = SentenceTransformer(model, device=device)
#         else:
#             self.model = model

#     def encode(self, docs: List[str]) -> np.ndarray:
#         embeddings = self.model.encode(
#             docs,
#             show_progress_bar=self.verbose,
#             normalize_embeddings=self.normalize_embeddings
#         )
#         return embeddings