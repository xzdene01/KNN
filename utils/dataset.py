import h5py
import torch
import numpy as np


def get_shuffled_idxs(num_total, num_samples, device="cpu"):
    idxs = np.random.choice(num_total, size=num_samples, replace=False)
    idxs = np.sort(idxs) # h5py requires sorted indices
    return torch.from_numpy(idxs).to(device)

def load_h5(h5_path, device="cpu"):
    with h5py.File(h5_path, "r") as h5_file:
        dset = h5_file["embeddings"]
        embeddings = torch.tensor(dset).to(device)
    return embeddings

def save_h5(h5_path, embeddings):
    with h5py.File(h5_path, "w") as h5_file:
        h5_file.create_dataset("embeddings", data=embeddings)