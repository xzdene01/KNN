from wrappers.fastopic_wrapper import FASTopicWrapper
from wrappers.wrapper_base import WrapperBase
from utils.dataset import load_h5
from topmost import eva
from topmost import RawDataset, FASTopicTrainer
import pandas as pd
from stop_words import get_stop_words
from utils.tokenizers import CzechLemmatizedTokenizer
from topmost import Preprocess
import numpy as np
from wrappers.lda_wrapper import LDAWrapper
import stopwordsiso as stopwords

# Wrapper class for evaluating the five evaluation metrics of a model
# 
# model_wrapper is a FASTopicWrapper - model we are evaluating
# 
# test_data_path is the path to the CSV file with test data, this data
# is used for calculating NMI, Purity and classifier performance
#
# test_dataset_embeddings_path is the h5 file with embeddings
class EvaluationWrapper:
    def __init__(self, model_wrapper: WrapperBase):
        self.model_wrapper = model_wrapper
        self.args = model_wrapper.args

        if self.args.test_docs_path is None:
            raise ValueError("test_docs_path is required for evaluation.")
        
        if self.args.test_embes_path is None:
            raise ValueError("test_embes_path is required for evaluation.")

        self.test_dataset_path = self.args.test_docs_path
        self.test_dataset_embeddings_path = self.args.test_embes_path

    def evaluate(self):
        top_words = self.model_wrapper.model.get_top_words(self.model_wrapper.args.num_top_words, verbose=False)
        diversity = eva.topic_diversity._diversity(top_words)

        texts = self.model_wrapper.all_docs
        vocab = self.model_wrapper.model.vocab
        top_words = self.model_wrapper.model.get_top_words(self.model_wrapper.args.num_top_words, verbose=False)
        stop_words = get_stop_words(self.args.stopwords) + list(stopwords.stopwords("cs"))

        # Fixes issue with nan: https://github.com/BobXWu/TopMost/issues/12
        tokenizer = CzechLemmatizedTokenizer(stopwords=stop_words, cache_dir=self.args.cache_dir)
        preprocessor = Preprocess(tokenizer=tokenizer, vocab_size = self.args.vocab_size, stopwords=stop_words, seed=self.args.seed, verbose=self.args.verbose)
        preprocessed_docs, _ = preprocessor.parse(texts, vocab=vocab)

        coherence = eva.topic_coherence._coherence(preprocessed_docs, vocab, top_words)

        # Loading of dataset for clustering and classification
        dataset = pd.read_csv(self.test_dataset_path)
        test_data = dataset["content"]
        test_labels = dataset["topic"]
        n_topics = test_labels.nunique()
        dataset_embeddings = load_h5(self.test_dataset_embeddings_path, device=self.args.device)

        # Calculate Purity and NMI
        test_theta = self.model_wrapper.model.transform(test_data, dataset_embeddings)
        clustering_results = eva._clustering(test_theta, test_labels)

        # Split dataset into training and testing portions
        train_dataset, test_dataset, train_embeds, test_embeds = split_test_train(dataset, dataset_embeddings)

        # Calculate accuracy and F1
        test_theta = self.model_wrapper.model.transform(test_dataset["content"], test_embeds)
        train_theta = self.model_wrapper.model.transform(train_dataset["content"], train_embeds)

        classification_results = eva.classification._cls(train_theta, test_theta, train_dataset["topic"], test_dataset["topic"])
       
        return {"coherence": coherence, "topic_diversity": diversity, "purity": clustering_results["Purity"], "nmi": clustering_results["NMI"], "accuracy": classification_results["acc"], "f1-score": classification_results["macro-F1"]}

# Splits the dataset into a training samples and testing samples
# 
# dataset = pandas dataframe of the loaded dataset with "content" and "topic" (labels) columns
# embeddings = embeddings loaded from the h5 file
# ratio = ratio of testing to training samples
def split_test_train(dataset, embeddings, ratio=0.1):
    assert(len(dataset) == len(embeddings))

    dataset_len = len(dataset)
    train_len = int(dataset_len * (1 - ratio))

    train_dataset = dataset[0:train_len]
    train_embeds = embeddings[0:train_len]

    test_dataset = dataset[train_len:]
    test_embeds = embeddings[train_len:]

    return train_dataset, test_dataset, train_embeds, test_embeds

# Seperated class for LDA
class LDAEvaluationWrapper:
    def __init__(self, model_wrapper: LDAWrapper):
        self.model_wrapper = model_wrapper
        self.args = model_wrapper.args

        if self.args.test_docs_path is None:
            raise ValueError("test_docs_path is required for evaluation.")

        self.test_dataset_path = self.args.test_docs_path
        self.dictionary = model_wrapper.dictionary  # Gensim dictionary from wrapper
        self.preprocessor = self.create_preprocessor()

    def get_top_words(self, words=15):
        topics = self.model_wrapper.model.show_topics(num_topics=-1,
            num_words=words,
            formatted=False
        )
        top_words = []
        for _, word_tuples in topics:
            words = [word for word, _ in word_tuples]
            top_words.append(words)
        
        return top_words
    
    def create_preprocessor(self):
        stop_words = get_stop_words(self.args.stopwords)
        tokenizer = CzechLemmatizedTokenizer(
            stopwords=stop_words,
            cache_dir=self.args.cache_dir
        )
        return Preprocess(
            tokenizer=tokenizer,
            vocab_size=self.args.vocab_size,
            stopwords=stop_words,
            seed=self.args.seed,
            verbose=self.args.verbose
        )
    
    # Convert Gensim theta format to numpy array.
    def theta_to_numpy(self, theta):
        n_topics = self.model_wrapper.model.num_topics
        theta_np = np.zeros((len(theta), n_topics))
        for i, doc in enumerate(theta):
            for topic_id, prob in doc:
                theta_np[i, topic_id] = prob
        return theta_np
    
    # Convert texts to BoW and topic distributions.
    def load_and_preprocess_data(self, data):
        preprocessed_docs, _ = self.preprocessor.parse(data, vocab=self.model_wrapper.vocab)
        test_docs = []
        for doc in preprocessed_docs:
            # To ensure list of tokens - without this part the output from tokenizer was not enough to convert to Bowl of words
            if isinstance(doc, str):
                doc = doc.split()
    
            # Now convert to Bowl of words
            bow = self.model_wrapper.dictionary.doc2bow(doc)
            test_docs.append(bow)
        theta = self.model_wrapper.model[test_docs]
        return self.theta_to_numpy(theta)
    
    def evaluate(self):
        top_words = self.get_top_words(self.args.num_top_words)
        topics_as_strings = [' '.join(words) for words in top_words]
        
        # Coherence and Diversity calculation
        texts = self.model_wrapper.all_docs
        preprocessed_docs, _ = self.preprocessor.parse(texts, vocab=self.model_wrapper.vocab)
        coherence = eva.topic_coherence._coherence(preprocessed_docs, self.model_wrapper.vocab, topics_as_strings)
        diversity = eva.topic_diversity._diversity(topics_as_strings)

        # Load labeled dataset
        dataset = pd.read_csv(self.test_dataset_path)
        test_data = dataset["content"].astype(str).tolist()
        test_labels = dataset["topic"]

        # Clustering Metrics (Purity/NMI)
        test_theta_np = self.load_and_preprocess_data(test_data)
        clustering_results = eva._clustering(test_theta_np, test_labels)

        # Classification Metrics (Accuracy/F1)
        train_dataset, test_dataset = self._split_test_train(dataset)
        train_data = train_dataset["content"].astype(str).tolist()
        test_data = test_dataset["content"].astype(str).tolist()

        train_theta_np = self.load_and_preprocess_data(train_data)
        test_theta_np = self.load_and_preprocess_data(test_data)
        
        classification_results = eva.classification._cls(
            train_theta_np, 
            test_theta_np, 
            train_dataset["topic"], 
            test_dataset["topic"]
        )

        return {
            "coherence": coherence,
            "topic_diversity": diversity,
            "purity": clustering_results["Purity"],
            "nmi": clustering_results["NMI"],
            "accuracy": classification_results["acc"],
            "f1-score": classification_results["macro-F1"]
        }

    def _split_test_train(self, dataset, ratio=0.1):
        """Split dataset without embeddings"""
        dataset_len = len(dataset)
        train_len = int(dataset_len * (1 - ratio))
        return dataset.iloc[:train_len], dataset.iloc[train_len:]
