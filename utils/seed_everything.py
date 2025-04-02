import os
import torch
import random
import argparse
import numpy as np
import logging

def seed_everything(args: argparse.Namespace):
    if args.seed is None:
        args.seed = random.randint(0, 2**32 - 1)

    seed = args.seed
    # Seed Python's random module
    random.seed(seed)
    
    # Seed NumPy's random generator
    np.random.seed(seed)
    
    # Seed PyTorch's CPU RNG
    torch.manual_seed(seed)
    
    # Seed all available GPUs (if any)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # For cudnn to be deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

    logging.info(f"Seed set to {args.seed}.")