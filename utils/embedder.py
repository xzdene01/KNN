# Part of this file is copied from Ing. Martin Fajčík Ph.D. and slightly modified

import torch
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


class BasicSentenceEmbedder:
    def __init__(
        self,
        model,
        device: str="cpu",
        normalize_embeddings: bool = False,
        verbose: bool = False,
        torch_dtype: torch.dtype = None
    ):
        self.verbose = verbose
        self.normalize_embeddings = normalize_embeddings

        if isinstance(model, str):
            if torch_dtype is None:
                self.model = SentenceTransformer(model, device=device)
            else:
                self.model = SentenceTransformer(model, device=device, model_kwargs={"torch_dtype": torch_dtype})
        else:
            self.model = model

    def encode(self, docs: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            docs,
            # show_progress_bar=self.verbose,
            normalize_embeddings=self.normalize_embeddings
        )
        return embeddings
