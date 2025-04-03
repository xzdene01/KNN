from wrappers.fastopic_wrapper import FASTopicWrapper
from topmost.eva import topic_diversity, topic_coherence

# Wrapper class for evaluating the five evaluation metrics of a model
# model_wrapper is a FASTopicWrapper - model we are evaluating
# test_data is 
class EvaluationWrapper:
    def __init__(self, model_wrapper):
        self.model_wrapper = model_wrapper
        # self.test_data = test_data

    def evaluate(self):
        top_words = self.model_wrapper.model.get_top_words(self.model_wrapper.args.num_top_words, verbose=False)
        diversity = topic_diversity._diversity(top_words)

        texts = self.model_wrapper.all_docs
        vocab = self.model_wrapper.model.vocab
        top_words = self.model_wrapper.model.get_top_words(self.model_wrapper.args.num_top_words, verbose=False)
        coherence = topic_coherence._coherence(texts, vocab, top_words)

        # For purity, NMI and classifier - I need to load a labeled dataset, probably a good idea to store it in this wrapper?
        #
        # Problems:
        # - How to load this data? Where is that data from?
        # - Converting between Topmost RawDatasetHandler class and FASTopic dataset

        # I can get theta from the the model:
        # theta = self.model_wrapper.model.train_theta

        # How to perform these evaluations: https://topmost.readthedocs.io/en/stable/quick_start.html#evaluate
        # 
        # # get theta (doc-topic distributions)
        # train_theta, test_theta = trainer.export_theta(dataset)
        # # evaluate clustering
        # clustering_results = topmost.evaluations.evaluate_clustering(test_theta, dataset.test_labels)
        # # evaluate classification
        # classification_results = topmost.evaluations.evaluate_classification(train_theta, test_theta, dataset.train_labels, dataset.test_labels)
       
        return {"coherence": coherence, "topic_diversity": diversity}
