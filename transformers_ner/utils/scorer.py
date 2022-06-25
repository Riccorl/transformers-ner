from typing import Dict, List, Union
from datasets import load_metric

from data.labels import Labels


class SeqevalScorer:
    def __init__(self):
        self.metric = load_metric("seqeval")

    def __call__(
        self,
        predictions: Union[List[str], List[List[str]]],
        references: Union[List[str], List[List[str]]],
    ) -> Dict:
        """
        Args:
            predictions (:obj:`List[str]`, :obj:`List[List[str]]`):
                The predictions of the model.
            references (:obj:`List[str]`, :obj:`List[List[str]]`):
                The ground truth.
        Returns:
            :obj:`Dict`: a dictionary containing the name of the metrics and their values
        """
        return self.metric.compute(predictions=predictions, references=references)
