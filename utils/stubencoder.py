# This file if copied from Ing. Martin Fajčík Ph.D. and slightly modified

from typing import List


class StubDocEncoderFastTopic:
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def encode(self,
               docs: List[str],
               show_progress_bar: bool = False,
               normalize_embeddings: bool = False
               ):
        assert len(docs) == self.embeddings.shape[0]
        return self.embeddings
