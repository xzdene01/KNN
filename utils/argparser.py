import logging
import argparse

def get_log_level(args: argparse.Namespace):
    if args.debug:
        return logging.DEBUG
    elif args.verbose:
        return logging.INFO
    else:
        return logging.WARNING

def get_args():
    parser = argparse.ArgumentParser(description='FASTopic wrapper for training, evaluation and comparision with LDA and BERTopic.')

    parser.add_argument("--seed", default=None, type=int, help="Number to use for seeding all random actions (default: use random seed).")
    parser.add_argument("--cache_dir", default="./models", type=str, help="Directory to store the downloaded models (not for final FASTopic model).")

    parser.add_argument('--load_path', type=str, help="Path to pretrained model checkpoint (default: no action).")
    parser.add_argument('--save_path', type=str, help="Path to save the model to (default: no action).")

    # Preprocessing parameters
    parser.add_argument("--vocab_size", type=int, default=None, help="Size of vocabulary (default: no limit).")
    parser.add_argument("--stopwords",  type=str, default="cz",   help="Language of stopwords (default: cz).")

    # Documents parameters
    parser.add_argument("--docs_path",  type=str, required=True,    help="Path to documents file used for training and unsupervised tests.")
    parser.add_argument("--num_docs",   type=int, default=15_000,     help="Number of documents and/or embeddings to use (default: 15k).")

    # Embeddings parameters
    parser.add_argument("--embes_path", type=str, default=None, help="Path to save/load h5 embeddings (default: compute embeddings on the fly).")
    parser.add_argument("--embe_model", type=str, default="all-MiniLM-L6-v2", help="Model to use for embeddings (default: all-MiniLM-L6-v2).")

    # FASTopic model parameters (will be ignored if loading model)
    parser.add_argument("--num_topics",     type=int,    default=50,    help="Number of topics (default: 50).")
    parser.add_argument("--num_top_words",  type=int,    default=15,    help="Number of top words per topic (default: 15).")
    parser.add_argument("--device",         type=str,    default=None,  help="Device to use for training (default: auto try cuda).")
    parser.add_argument("--log_interval",   type=int,    default=10,    help="Log interval when training (default: 20).")
    parser.add_argument("--batch_size",     type=int,    default=None,  help="Batch size for low memory mode (default: all docs).")
    parser.add_argument("--norm_embes", action='store_true', help="Normalize embeddings, this will be passed to document embedder.")

    # Training parameters (will be ignored if loading model)
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs to train (default: 200).")
    parser.add_argument("--lr",     type=float, default=0.002, help="Learning rate (default: 0.002).")

    parser.add_argument("--log_path", type=str, default=None, help="Path to save logs (default: no action).")
    parser.add_argument("--eval_dir", type=str, default=None, help="Path to save evaluation results (default: no action).")

    # Choose just verbore or debug, if debug set verbore to True
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--verbose", action='store_true', help="Print additional info.")
    group.add_argument("--debug",   action='store_true', help="Print even more debug info.")

    args = parser.parse_args()

    if args.debug:
        args.verbose = True

    return args
