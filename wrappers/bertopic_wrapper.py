from utils.dataset import load_docs
from stop_words import get_stop_words
import stopwordsiso as stopwords
from utils.tokenizers import CzechLemmatizedTokenizer
from topmost.preprocess import Preprocess
import logging
from wrappers.wrapper_base import WrapperBase
from bertopic import BERTopic
from topmost.data.basic_dataset import RawDataset
from utils.embedder import StubEncoder
from utils.dataset import load_h5,get_shuffled_idxs

class BERTopicWrapper():
    def __init__(self, args):
        self.args = args

        # Load docs from jsonl file (jsonl or csv)
        docs = load_docs(args.docs_path)
        if args.num_docs > len(docs):
            args.num_docs = len(docs)
            logging.warning(f"Number of documents is smaller than requested number of documents, using {args.num_docs} documents.")
        self.doc_idxs = get_shuffled_idxs(len(docs), args.num_docs, device=args.device)
        self.all_docs = docs
        docs = [docs[i] for i in self.doc_idxs]

        # Load training text embeddings
        embeddings = load_h5(args.embes_path, device=args.device)
        embedder = StubEncoder(embeddings)

        # Either train the model on the embeddings or load it from a file
        if args.load_path:
            logging.info(f"Loading BERTopic model from {args.load_path}")
            self.model = BERTopic.load(args.load_path, embedding_model=embedder)
        else:
            logging.info(f"Creating BERTopic model")
            self.model = BERTopic(language="multilingual",
                top_n_words=args.num_top_words,
                nr_topics=args.num_topics,
                embedding_model=embedder
            )
            logging.info(f"Training BERTopic model")
            self.model.fit(self.all_docs, embeddings=embeddings.numpy())
            logging.info(f"BERTopic model trained successfully")

            # Optionally save the trained model
            if args.save_path:
                self.model.save(args.save_path)
                logging.info(f"Model saved to {args.save_path}.")
            else:
                logging.warning("Model was not saved, use --save_path to save the model.")
