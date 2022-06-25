from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import transformers_embedder as tre
from data.labels import Labels

from models.viterbi import ViterbiDecoder


class TransformersNER(torch.nn.Module):
    def __init__(
        self,
        labels: Labels,
        language_model: str = "bert-base-cased",
        layer_pooling_strategy: str = "mean",
        subword_pooling_strategy: str = "scatter",
        fine_tune: bool = True,
        dropout: float = 0.2,
        projection_size: int = 512,
        use_viterbi: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        # labels
        self.labels = labels
        # transformer layer
        self.transformer = tre.TransformersEmbedder(
            language_model,
            subword_pooling_strategy=subword_pooling_strategy,
            layer_pooling_strategy=layer_pooling_strategy,
            fine_tune=fine_tune,
            *args,
            **kwargs,
        )
        self.linears = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(
                self.transformer.hidden_size,
                projection_size,
            ),
            torch.nn.ReLU(),
        )
        # classifier
        self.classifier = torch.nn.Linear(projection_size, self.labels.get_label_size())
        # output viterbi decoder
        self.viterbi_decoder = ViterbiDecoder(self.labels) if use_viterbi else None

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sparse_offsets: Optional[torch.Tensor] = None,
        scatter_offsets: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        compute_loss: bool = False,
        compute_predictions: bool = False,
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            input_ids (`torch.Tensor`):
                The input ids of the sentences.
            attention_mask (`torch.Tensor`):
                The attention mask of the sentences.
            sparse_offsets (`torch.Tensor`):
                The token type ids of the sentences.
            scatter_offsets (`torch.Tensor`):
                The offsets of the sentences.
            labels (`torch.Tensor`):
                The labels of the sentences.
            compute_predictions (`bool`):
                Whether to compute the predictions.
            compute_loss (`bool`):
                Whether to compute the loss.

        Returns:
            obj:`torch.Tensor`: The outputs of the model.
        """
        model_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "sparse_offsets": sparse_offsets,
            "scatter_offsets": scatter_offsets,
        }
        embeddings = self.transformer(**model_kwargs).word_embeddings
        logits = self.linears(embeddings)
        logits = self.classifier(logits)

        output = {"logits": logits}

        if compute_predictions:
            if self.viterbi_decoder:
                predictions = [self.viterbi_decoder(l) for l in logits.cpu()]
                predictions = [pred[0] for pred in predictions]
            else:
                predictions = logits.argmax(dim=-1)
            output["predictions"] = predictions

        if compute_loss and labels is not None:
            output["loss"] = self.compute_loss(logits, labels)

        return output

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss of the model.

        Args:
            logits (`torch.Tensor`):
                The logits of the model.
            labels (`torch.Tensor`):
                The labels of the model.

        Returns:
            obj:`torch.Tensor`: The loss of the model.
        """
        return F.cross_entropy(
            logits.view(-1, self.labels.get_label_size()), labels.view(-1)
        )
