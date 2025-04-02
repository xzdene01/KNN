import os
import json
import logging
import argparse
from stop_words import get_stop_words
from topmost.preprocess import Preprocess
from fastopic import FASTopic
from fastopic._utils import DocEmbedModel
from topmost.eva import topic_diversity, topic_coherence

from utils.embedder import StubEncoder
from utils.dataset import get_shuffled_idxs, load_h5, save_h5
from utils.tokenizers import CzechLemmatizedTokenizer


class FASTopicWrapper:
    def __init__(self, args: argparse.Namespace):
        self.args = args

        # Load docs from jsonl file
        with open(args.docs_path, "r") as f:
            docs = [json.loads(line)["split"] for line in f.readlines()]
            
            if args.num_docs > len(docs):
                args.num_docs = len(docs)
                logging.warning(f"Number of documents is smaller than requested number of documents, using {args.num_docs} documents.")

            self.doc_idxs = get_shuffled_idxs(len(docs), args.num_docs, device=args.device)
            self.all_docs = docs
            docs = [docs[i] for i in self.doc_idxs]

        # !!! Hardcoded embedder name !!!
        self.embedder_name = "all-MiniLM-L6-v2"
        # self.embedder_name = "BAAI/bge-multilingual-gemma2"

        # Load or train FASTopic model
        if args.load_path:
            logging.info(f"Loading model from {args.load_path}.")
            self.model = FASTopic.from_pretrained(args.load_path)
        else:
            logging.info("Training model from scratch.")
        
            # Create Preprocessor that will be later used to convert data into BoW representation
            stop_words = get_stop_words(args.stopwords) # + get_stopwords()
            tokenizer = CzechLemmatizedTokenizer(stopwords=stop_words, cache_dir=args.cache_dir)
            preprocessor = Preprocess(tokenizer=tokenizer, vocab_size = args.vocab_size, stopwords=stop_words, seed=args.seed, verbose=args.verbose)

            if args.embes_path:
                # Load pre-computed embeddings from h5 file
                if os.path.exists(args.embes_path):
                    embeddings = load_h5(args.embes_path, device=args.device)
                    embeddings = embeddings[self.doc_idxs]
                    doc_embedder = StubEncoder(embeddings)
                    logging.info(f"Loaded {embeddings.shape[0]} embeddings from {args.embes_path} and initialized a stub encoder.")
                
                # Pre-compute embeddings and save them, then use stub encoder with them
                else:
                    embedder = DocEmbedModel(model=self.embedder_name,
                                             device=args.device,
                                             normalize_embeddings=args.norm_embes,
                                             verbose=args.verbose)
                    embeddings = embedder.encode(self.all_docs)
                    save_h5(args.embes_path, embeddings)
                    doc_embedder = StubEncoder(embeddings)
                    logging.info(f"Computed and saved {embeddings.shape[0]} embeddings to {args.embes_path}.")
            
            # Compute embeddings on the fly
            else:
                doc_embedder = self.embedder_name
                logging.info(f"Using {self.embedder_name} to compute embeddings on the fly.")

            # Create FASTopic model
            model = FASTopic(num_topics=args.num_topics,
                             preprocess=preprocessor,
                             doc_embed_model=doc_embedder,

                             num_top_words=args.num_top_words,
                             device=args.device,
                             log_interval=args.log_interval,
                             low_memory=args.batch_size is not None,
                             low_memory_batch_size=args.batch_size,
                             verbose=args.verbose,
                             normalize_embeddings=args.norm_embes)
            
            model.fit_transform(docs, epochs=args.epochs, learning_rate=args.lr)
            self.model = model
        
        # Save model
        if args.save_path:
            try:
                self.model.save(args.save_path)
            except Exception as e:
                logging.warning(f"Failed to save model to {args.save_path}. Saving model without tokenizer.")
                del self.model.preprocess.tokenizer
                self.model.save(args.save_path)

            logging.info(f"Model saved to {args.save_path}.")
        elif not args.load_path:
            logging.warning("Model was not saved, use --save_path to save the model.")
    
    @property
    def topic_diversity(self):
        top_words = self.model.get_top_words(self.args.num_top_words, verbose=False)
        return topic_diversity._diversity(top_words)
    
    @property
    def topic_coherence(self):
        texts = self.all_docs
        vocab = self.model.vocab
        top_words = self.model.get_top_words(self.args.num_top_words, verbose=False)
        coherence = topic_coherence._coherence(texts, vocab, top_words)
        return coherence
    
    def visualize_hierarchy(self):
        fig = self.model.visualize_topic_hierarchy()
        fig.show()
    
    def visualize_weights(self):
        fig = self.model.visualize_topic_weights()
        fig.show()