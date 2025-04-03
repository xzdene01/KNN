import os
os.environ["HF_HOME"] = "./models"

import torch
from sentence_transformers import SentenceTransformer

from utils.dataset import load_docs, save_h5


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load documents from a file
    docs_path = "data/reduced_dataset.csv"
    docs = load_docs(docs_path)
    print(f"Loaded {len(docs)} documents from {docs_path}")

    # Load model
    # model_name = "paraphrase-multilingual-MiniLM-L12-v2"
    # model = SentenceTransformer(model_name, device=device)
    model_name = "BAAI/bge-multilingual-gemma2"
    model = SentenceTransformer(model_name, device=device, model_kwargs={"torch_dtype": torch.float16})
    print(f"Loaded model: {model_name}")

    # Encode documents
    embeddings = model.encode(docs, show_progress_bar=True, normalize_embeddings=False)
    print(f"Encoded {len(docs)} documents")

    # Save embeddings to HDF5 file
    h5_path = "data/reduced_dataset_bge-multilingual-gemma2.h5"
    save_h5(h5_path, embeddings)


if __name__ == '__main__':
    main()