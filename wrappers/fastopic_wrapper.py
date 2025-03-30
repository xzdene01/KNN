import h5py
import json
import argparse
import logging
from stop_words import get_stop_words
from topmost.preprocess import Preprocess
from fastopic import FASTopic

from topmost.eva import topic_diversity

from utils.stubencoder import StubDocEncoderFastTopic

class FASTopicWrapper:
    def __init__(self, args: argparse.Namespace):
        self.args = args

        # Load or train FASTopic model
        if args.load_path:
            logging.info(f"Loading model from {args.load_path}")

            self.model = FASTopic.from_pretrained(args.load_path)
        else:
            logging.info("Training model from scratch")
        
            # Create Preprocessor that will be later used to convert data into BoW representation
            stop_words = get_stop_words(args.stopwords)
            preprocessor = Preprocess(vocab_size = args.vocab_size, stopwords=stop_words)

            # Create DocEmbedder that will either process texts to embeddings or load precomputed embeddings
            if args.embes_path:
                h5_file = h5py.File(args.embes_path, "r")
                embeddings = h5_file["embeddings"]
                embeddings = embeddings[:args.num_docs] if args.num_docs else embeddings
                doc_embedder = StubDocEncoderFastTopic(embeddings)
            else:
                raise NotImplementedError("Computing embeddings on the fly is not implemented yet.")

            # Create FASTopic model
            model = FASTopic(num_topics=args.num_topics,
                             num_top_words=args.num_top_words,
                             device=args.device,
                             log_interval=args.log_interval,
                             low_memory=args.batch_size is not None,
                             low_memory_batch_size=args.batch_size,
                             verbose=args.verbose,
                             normalize_embeddings=args.norm_embes,
                             preprocess=preprocessor, doc_embed_model=doc_embedder)
            
            # Load docs from jsonl file
            with open(args.docs_path, "r") as f:
                docs = [json.loads(line)["split"] for line in f.readlines()]
                docs = docs[:args.num_docs] if args.num_docs else docs
            model.fit_transform(docs, epochs=args.epochs, learning_rate=args.lr)

            self.model = model
        
        # Save model if requested
        if args.save_path:
            logging.info(f"Saving model to {args.save_path}")
            self.model.save(args.save_path)
    
    @property
    def topic_diversity(self):
        return topic_diversity._diversity(self.model.get_top_words(self.args.num_top_words,self.args.verbose))