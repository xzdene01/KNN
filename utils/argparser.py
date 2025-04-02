import argparse

def get_args():
    parser = argparse.ArgumentParser(description='FASTopic wrapper for training, evaluation and comparision with LDA and BERTopic.')

    parser.add_argument("--seed", default=42, type=int, help="Number to use for seeding all random actions (default: 42).")
    parser.add_argument("--cache_dir", default="./models", type=str, help="Directory to store the downloaded models (not for final FASTopic model).")

    parser.add_argument('--load_path', type=str, help="Path to pretrained model checkpoint (default: no action).")
    parser.add_argument('--save_path', type=str, help="Path to save the model to (default: no action).")

    # Preprocessing parameters
    parser.add_argument("--vocab_size", type=int, default=7_000, help="Size of vocabulary (default: 7k).")
    parser.add_argument("--stopwords",  type=str, default="cz",   help="Language of stopwords (default: cz).")

    # Documents parameters
    parser.add_argument("--docs_path",  type=str, default=None, help="Path to documents file (required if not loading model).")
    parser.add_argument("--num_docs",   type=int, default=None, help="Number of documents and/or embeddings to use (default: all, but thats way too much).")

    # Embeddings parameters
    parser.add_argument("--embes_path", type=str, default=None, help="Path to 5h embeddings file (default: compute embeddings on the fly).")

    # FASTopic model parameters (will be ignored if loading model)
    parser.add_argument("--num_topics",     type=int,    default=50,    help="Number of topics (default: 50).")
    parser.add_argument("--num_top_words",  type=int,    default=15,    help="Number of top words per topic (default: 15).")
    parser.add_argument("--device",         type=str,    default=None,  help="Device to use for training (default: auto try cuda).")
    parser.add_argument("--log_interval",   type=int,    default=10,    help="Log interval when training (default: 20).")
    parser.add_argument("--batch_size",     type=int,    default=None,  help="Batch size for low memory mode (default: all docs).")

    parser.add_argument("--norm_embes", action='store_true', help="Normalize embeddings, this will be passed to document embedder.")
    parser.add_argument("--verbose",    action='store_true', help="Print additional info during training.")

    # Training parameters (will be ignored if loading model)
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs to train (default: 300).")
    parser.add_argument("--lr",     type=float, default=0.002, help="Learning rate (default: 0.002).")

    # Eval parameters
    parser.add_argument("--eval", action='store_true', help="Run evaluation on the model.")

    args = parser.parse_args()
    return args
