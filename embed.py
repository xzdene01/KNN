import os
os.environ["HF_HOME"] = "./models"

import torch
import numpy as np
from sentence_transformers import SentenceTransformer

from utils.dataset import load_docs, save_h5

base_path = "data/reduced_dataset"
model_name = "BAAI/bge-multilingual-gemma2"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load documents from a file
    docs_path = base_path + ".csv"
    docs = load_docs(docs_path)
    print(f"Loaded {len(docs)} documents from {docs_path}")

    # Load model
    model = SentenceTransformer(model_name, device=device, model_kwargs={"torch_dtype": torch.float16})
    print(f"Loaded model: {model_name}")

    # Set max token length
    # model.max_seq_length = 1024

    # Encode documents
    embeddings = model.encode(docs, batch_size=1, show_progress_bar=True, normalize_embeddings=True)
    print(f"Encoded {len(docs)} documents")

    # check and cast to float32
    if embeddings.dtype != np.float32:
        print(f"Converting embeddings from {embeddings.dtype} to float32")
        embeddings = embeddings.astype(np.float32)

    # Save embeddings to HDF5 file
    h5_path = base_path + "_" + model_name + "_norm" + ".h5"
    save_h5(h5_path, embeddings)

    return


if __name__ == '__main__':
    main()
