import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import transformers_embedder as tre
from torch import nn
from torch.optim import RAdam
from torchmetrics.classification import F1


class NERModule(pl.LightningModule):
    def __init__(self, labels, *args, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.labels = labels
        # layers
        self.language_model = tre.TransformersEmbedder(
            self.hparams.language_model_name,
            return_words=self.hparams.return_words,
            output_layer=self.hparams.output_layer,
            fine_tune=self.hparams.lm_fine_tune,
        )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.language_model.hidden_size, len(self.labels), bias=False)
        # metrics
        self.f1 = F1(len(self.labels))

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        offsets: torch.Tensor = None,
        *args,
        **kwargs
    ) -> torch.Tensor:
        x = self.language_model(input_ids, offsets, attention_mask, token_type_ids).word_embeddings
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, f1_score = self.shared_step(batch)
        log = {"loss": loss, "f1": f1_score}
        self.log_dict(log)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        loss, f1_score = self.shared_step(batch)
        log = {"val_loss": loss, "val_f1": f1_score}
        self.log_dict(log)

    def test_step(self, batch: dict, batch_idx: int) -> None:
        loss, f1_score = self.shared_step(batch)
        log = {"test_loss": loss, "test_f1": f1_score}
        self.log_dict(log)

    def shared_step(self, batch: dict):
        x, y = batch
        y_hat = self.forward(**x)
        loss = F.cross_entropy(y_hat.view(-1, len(self.labels)), y.view(-1))
        y_hat = torch.argmax(y_hat, dim=-1)
        f1_score = []
        for i, sentence_length in enumerate(x["sentence_length"]):
            f1_score.append(self.f1(y_hat[i, :sentence_length], y[i, :sentence_length]))
        f1_score = sum(f1_score) / len(f1_score)
        return loss, f1_score

    def configure_optimizers(self):
        groups = [
            {
                "params": self.classifier.parameters(),
                "lr": self.hparams.lr,
            },
            {
                "params": self.language_model.parameters(),
                "lr": self.hparams.lm_lr,
                "correct_bias": False,
            },
        ]
        return RAdam(groups)
