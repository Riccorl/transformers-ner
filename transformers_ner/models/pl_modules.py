from typing import Dict, List, Union

import hydra
import pytorch_lightning as pl
import torch
from torch.optim import RAdam
from transformers_embedder.tokenizer import ModelInputs

from data.labels import Labels
from utils.scorer import SeqevalScorer


class NERModule(pl.LightningModule):
    def __init__(self, labels: Labels, *args, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.labels = labels
        # model
        self.model = hydra.utils.instantiate(self.hparams.model, labels=labels)
        # metrics
        self.seqeval_scorer = SeqevalScorer()

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Returns:
            obj:`torch.Tensor`: The outputs of the model.
        """
        return self.model(**kwargs)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        # training kwargs
        training_kwargs = {**batch, "compute_loss": True}
        outputs = self.forward(**training_kwargs)
        self.log("loss", outputs["loss"])
        return outputs["loss"]

    def validation_step(self, batch: ModelInputs, batch_idx: int) -> None:
        # val kwargs
        val_kwargs = {
            **batch,
            "compute_loss": True,
            "compute_predictions": True,
        }
        batch_size = len(batch.input_ids)
        # get output from model
        outputs = self.forward(**val_kwargs)
        self.log("val_loss", outputs["loss"], batch_size=batch_size)
        # compute f1 score
        metrics = self.compute_f1_score(
            outputs["predictions"], batch["labels"], batch["sentence_lengths"]
        )
        for metric_name, metric_value in metrics.items():
            if "overall_" in metric_name:
                self.log(
                    f"val_{metric_name}",
                    metric_value,
                    prog_bar=True,
                    batch_size=batch_size,
                )

    def test_step(self, batch: dict, batch_idx: int) -> None:
        # test kwargs
        test_kwargs = {
            **batch,
            "compute_loss": True,
            "compute_predictions": True,
        }
        batch_size = len(batch.input_ids)
        # get output from model
        outputs = self.forward(**test_kwargs)
        self.log("test_loss", outputs["loss"], batch_size=batch_size)
        # compute f1 score
        metrics = self.compute_f1_score(
            outputs["predictions"], batch["labels"], batch["sentence_lengths"]
        )
        for metric_name, metric_value in metrics.items():
            if "overall_" in metric_name:
                self.log(
                    f"test_{metric_name}",
                    metric_value,
                    prog_bar=True,
                    batch_size=batch_size,
                )

    def compute_f1_score(
        self,
        predictions: Union[List, torch.Tensor],
        labels: Union[List, torch.Tensor],
        sentence_lengths: List,
    ) -> Dict:
        # we need to convert them to strings
        # if it is a tensor, we need to convert it to a list
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().tolist()
        # then we retrieve the named labels
        predictions = [
            [self.labels.get_label_from_index(p) for p in preds[:length]]
            for preds, length in zip(predictions, sentence_lengths)
        ]
        # same for labels
        labels[labels == -100] = 0
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().tolist()
        labels = [
            [self.labels.get_label_from_index(l) for l in label[:length]]
            for label, length in zip(labels, sentence_lengths)
        ]
        # return scores
        metrics = self.seqeval_scorer(predictions, labels)
        return metrics

    def configure_optimizers(self):
        base_parameters = []
        lm_decay_parameters = []
        lm_no_decay_parameters = []

        for parameter_name, parameter in self.named_parameters():
            if "transformer" not in parameter_name:
                base_parameters.append(parameter)
            elif not any(v in parameter_name for v in ["bias", "LayerNorm.weight"]):
                lm_decay_parameters.append(parameter)
            else:
                lm_no_decay_parameters.append(parameter)

        optimizer_params = [
            {
                "params": base_parameters,
                "weight_decay": self.hparams.optim_params.weight_decay,
            },
            {
                "params": lm_decay_parameters,
                "lr": self.hparams.optim_params.lm_lr,
                "weight_decay": self.hparams.optim_params.lm_weight_decay,
            },
            {
                "params": lm_no_decay_parameters,
                "lr": self.hparams.optim_params.lm_lr,
                "weight_decay": 0.0,
            },
        ]

        optimizer = RAdam(optimizer_params, lr=self.hparams.optim_params.lr)
        return optimizer
