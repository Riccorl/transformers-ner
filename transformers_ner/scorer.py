from datasets import load_metric

from data.labels import Labels


class SeqevalScorer:
    def __init__(self, labels: Labels):
        self.scorer = load_metric("seqeval")
        self.labels = labels

    def __call__(self, predictions, golds, *args, **kwargs):
        true_predictions = []
        true_labels = []
        for prediction, label in zip(predictions, golds):
            true_predictions.append([self.labels.get_label_from_index(p) for p in prediction])
            true_labels.append([self.labels.get_label_from_index(l) for l in label])
        return self.scorer.compute(predictions=true_predictions, references=true_labels)
