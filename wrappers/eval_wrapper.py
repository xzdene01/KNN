from wrappers.fastopic_wrapper import FASTopicWrapper
from utils.dataset import load_h5
from topmost import eva
from topmost import RawDataset, FASTopicTrainer
import pandas as pd

# Wrapper class for evaluating the five evaluation metrics of a model
# 
# model_wrapper is a FASTopicWrapper - model we are evaluating
# 
# test_data_path is the path to the CSV file with test data, this data
# is used for calculating NMI, Purity and classifier performance
#
# test_dataset_embeddings_path is the h5 file with embeddings
class EvaluationWrapper:
    def __init__(self, model_wrapper, test_dataset_path, test_dataset_embeddings_path):
        self.model_wrapper = model_wrapper
        self.test_dataset_path = test_dataset_path
        self.test_dataset_embeddings_path = test_dataset_embeddings_path

    def evaluate(self):
        top_words = self.model_wrapper.model.get_top_words(self.model_wrapper.args.num_top_words, verbose=False)
        diversity = eva.topic_diversity._diversity(top_words)

        texts = self.model_wrapper.all_docs
        vocab = self.model_wrapper.model.vocab
        top_words = self.model_wrapper.model.get_top_words(self.model_wrapper.args.num_top_words, verbose=False)
        coherence = eva.topic_coherence._coherence(texts, vocab, top_words)

        # Loading of dataset for clustering and classification
        dataset = pd.read_csv(self.test_dataset_path)
        test_data = dataset["content"]
        test_labels = dataset["topic"]
        n_topics = test_labels.nunique()
        dataset_embeddings = load_h5(self.test_dataset_embeddings_path)

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
