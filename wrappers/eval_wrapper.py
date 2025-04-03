from wrappers.fastopic_wrapper import FASTopicWrapper
from topmost import eva
from topmost import RawDataset, FASTopicTrainer
import pandas as pd

# Wrapper class for evaluating the five evaluation metrics of a model
# 
# model_wrapper is a FASTopicWrapper - model we are evaluating
# 
# test_data_path is the path to the CSV file with test data, this data
# is used for calculating NMI, Purity and classifier performance
class EvaluationWrapper:
    def __init__(self, model_wrapper, test_dataset_path):
        self.model_wrapper = model_wrapper
        self.test_dataset_path = test_dataset_path

    def evaluate(self):
        top_words = self.model_wrapper.model.get_top_words(self.model_wrapper.args.num_top_words, verbose=False)
        diversity = eva.topic_diversity._diversity(top_words)

        texts = self.model_wrapper.all_docs
        vocab = self.model_wrapper.model.vocab
        top_words = self.model_wrapper.model.get_top_words(self.model_wrapper.args.num_top_words, verbose=False)
        coherence = eva.topic_coherence._coherence(texts, vocab, top_words)

        # Creation of new dataset for clustering and classification
        dataset = pd.read_csv(self.test_dataset_path)
        test_data = dataset["content"]
        test_labels = dataset["topic"]
        n_topics = test_labels.nunique()

        test_theta = self.model_wrapper.model.transform(test_data)
        clustering_results = eva._clustering(test_theta, test_labels)
        # Error in embedder
            # File "/home/robin/skola/sem8/knn/projekt/KNN/utils/embedder.py", line 18, in encode
            # assert len(docs) == self.embeddings.shape[0]
            # AssertionError

        # How to perform these evaluations: https://topmost.readthedocs.io/en/stable/quick_start.html#evaluate
        # 
        # # get theta (doc-topic distributions)
        # train_theta, test_theta = trainer.export_theta(dataset)
        # # evaluate clustering
        # clustering_results = topmost.evaluations.evaluate_clustering(test_theta, dataset.test_labels)
        # # evaluate classification
        # classification_results = topmost.evaluations.evaluate_classification(train_theta, test_theta, dataset.train_labels, dataset.test_labels)
       
        return {"coherence": coherence, "topic_diversity": diversity, "clusterings": clustering_results}
