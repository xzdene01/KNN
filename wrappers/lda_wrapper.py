from topmost.preprocess import Preprocess
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from utils.dataset import load_docs
from utils.tokenizers import CzechLemmatizedTokenizer
from stop_words import get_stop_words
import stopwordsiso as stopwords
from gensim.models import LdaMulticore

class LDAWrapper:
    def __init__(self, args):
        self.args = args
        self.all_docs = self._load_docs()
        
        # Initialize and run preprocessing
        stop_words = get_stop_words(args.stopwords) + list(stopwords.stopwords("cs"))
        tokenizer = CzechLemmatizedTokenizer(stopwords=stop_words, cache_dir=args.cache_dir)
        self.preprocessor = Preprocess(
            tokenizer=tokenizer,
            vocab_size=args.vocab_size,
            stopwords=stop_words,
            seed=args.seed,
            verbose=args.verbose
        )
        
        # Preprocess documents
        processed = self.preprocessor.preprocess(raw_train_texts=self.all_docs)
        
        self.vocab = processed['vocab']
        self.processed_texts = processed['train_texts']
        
        # Convert to Gensim format
        self.dictionary = Dictionary([text.split() for text in self.processed_texts])
        self.corpus = [self.dictionary.doc2bow(text.split()) for text in self.processed_texts]

        # Train LDA
        self.model = LdaMulticore(
            corpus=self.corpus,
            workers=4,
            num_topics=args.num_topics,
            id2word=self.dictionary,
            passes=10,              
            iterations=50,          
            alpha='symmetric',     
            chunksize=2000,          # Process documents in chunks
            random_state=args.seed,
            minimum_probability=0.0  # All topic probabilities
        )

    def _load_docs(self):
        return load_docs(self.args.docs_path)[:self.args.num_docs]