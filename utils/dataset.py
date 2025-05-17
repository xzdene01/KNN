import os
import h5py
import json
import torch
import pandas as pd
import numpy as np


def load_docs(path: str):
    extension = path.split(".")[-1]
    with open(path, "r") as f:
        if extension == "jsonl":
            return [json.loads(line)["split"] for line in f.readlines()]

        elif extension == "csv":
            df = pd.read_csv(f)
            return df["content"].tolist()

        else:
            raise ValueError(f"Unsupported file extension: {extension}. Supported extensions are: jsonl, csv.")


def get_shuffled_idxs(num_total, num_samples, device="cpu"):
    idxs = np.random.choice(num_total, size=num_samples, replace=False)
    idxs = np.sort(idxs)  # h5py requires sorted indices
    return torch.from_numpy(idxs).to(device)


def load_h5(h5_path, device="cpu"):
    with h5py.File(h5_path, "r") as h5_file:
        dset = h5_file["embeddings"]
        arr = np.array(dset)

    # assert that the on-disk dtype was float32
    assert arr.dtype == np.float32, (f"Expected embeddings in float32, but found {arr.dtype}.")

    # wrap in torch.Tensor (float32) and move to device
    emb = torch.from_numpy(arr).to(device)
    return emb


def save_h5(h5_path, embeddings: np.array):
    # ensure folder exists
    os.makedirs(os.path.dirname(h5_path), exist_ok=True)

    with h5py.File(h5_path, "w") as h5_file:
        h5_file.create_dataset("embeddings", data=embeddings, dtype=np.float32)
