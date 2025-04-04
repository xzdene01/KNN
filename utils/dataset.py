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
    idxs = np.sort(idxs) # h5py requires sorted indices
    return torch.from_numpy(idxs).to(device)


def load_h5(h5_path, device="cpu"):
    with h5py.File(h5_path, "r") as h5_file:
        dset = h5_file["embeddings"]
        dset = np.array(dset)
        embeddings = torch.tensor(dset).to(device)
    return embeddings.to(torch.float32)


def save_h5(h5_path, embeddings):
    with h5py.File(h5_path, "w") as h5_file:
        h5_file.create_dataset("embeddings", data=embeddings)