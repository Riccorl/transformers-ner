import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import transformer_embedder as tre
from torch import nn


class NERModule(pl.LightningModule):
    def __init__(self, conf, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(conf)
        # layers
        self.language_model = tre.TransformerEmbedder(
            conf.language_model,
            subtoken_pooling="mean",
            output_layer="sum",
            fine_tune=True,
        )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.language_model.hidden_size, 9, bias=False)
        # metrics
        self.f1 = pl.metrics.F1(9)

    def forward(self, inputs, *args, **kwargs) -> torch.Tensor:
        x = self.language_model(**inputs).word_embeddings
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
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat.view(-1, 9), y.view(-1))
        f1_score = self.f1(y_hat.view(-1, 9), y.view(-1))
        return loss, f1_score

    def configure_optimizers(self):
        groups = [
            {
                "params": self.classifier.parameters(),
                "lr": 2e-3,
            },
            {
                "params": self.language_model.parameters(),
                "lr": 1e-5,
                "correct_bias": False,
            },
        ]
        return torch.optim.Adam(groups)
