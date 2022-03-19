import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import transformers_embedder as tre
from torch import nn
from torch.optim import RAdam
from torchmetrics.classification import F1Score

from data.labels import Labels
from scorer import SeqevalScorer


class NERModule(pl.LightningModule):
    def __init__(self, labels: Labels, *args, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.labels = labels
        # layers
        self.language_model = tre.TransformersEncoder(
            self.hparams.language_model_name,
            return_words=self.hparams.return_words,
            layer_pooling_strategy=self.hparams.layer_pooling_strategy,
            fine_tune=self.hparams.lm_fine_tune,
        )
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(
            self.language_model.hidden_size, self.labels.get_label_size(), bias=False
        )
        # metrics
        self.seqeval_scorer = SeqevalScorer(self.labels)

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        offsets: torch.Tensor = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        x = self.language_model(input_ids, attention_mask, token_type_ids, offsets).word_embeddings
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, seqeval_scores = self.shared_step(batch)
        self.log("val_loss", loss)
        for metric_name, metric_value in seqeval_scores.items():
            if "overall_" in metric_name:
                self.log(metric_name, metric_value, prog_bar=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        loss, seqeval_scores = self.shared_step(batch)
        self.log("val_loss", loss)
        for metric_name, metric_value in seqeval_scores.items():
            if "overall_" in metric_name:
                self.log(f"val_{metric_name}", metric_value, prog_bar=True)

    def test_step(self, batch: dict, batch_idx: int) -> None:
        loss, seqeval_scores = self.shared_step(batch)
        self.log("test_loss", loss)
        for metric_name, metric_value in seqeval_scores.items():
            if "overall_" in metric_name:
                self.log(f"test_{metric_name}", metric_value, prog_bar=True)

    def shared_step(self, batch: dict):
        x, y = batch
        y_hat = self.forward(**x)
        loss = F.cross_entropy(y_hat.view(-1, self.labels.get_label_size()), y.view(-1))
        # compute seqeval metrics
        y_hat = torch.argmax(y_hat, dim=-1)
        predictions, labels = [], []
        for i, sentence_length in enumerate(x["sentence_lengths"]):
            predictions.append(y_hat[i, :sentence_length])
            labels.append(y[i, :sentence_length])
        seqeval_scores = self.seqeval_scorer(predictions, labels)
        return loss, seqeval_scores

    def configure_optimizers(self):
        groups = [
            {
                "params": self.classifier.parameters(),
                "lr": self.hparams.lr,
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": self.language_model.parameters(),
                "lr": self.hparams.lm_lr,
                "weight_decay": self.hparams.lm_weight_decay,
                "correct_bias": False,
            },
        ]
        return RAdam(groups)
