from typing import Dict, Optional

import torch
import transformers_embedder as tre
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from transformers_embedder.tokenizer import ModelInputs

from data.labels import Labels

from rich.console import Console
from rich.table import Table

console = Console()


class Dataset:
    def __init__(
        self,
        dataset_name: Optional[str] = None,
        max_length: int = 128,
        **kwargs,
    ):
        self.dataset_name = dataset_name
        self.max_length = max_length
        self.train_data = None
        self.dev_data = None
        self.test_data = None

    def print_stats(self):
        # print the number of samples in each dataset using rich tables
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Split", style="dim", width=12)
        table.add_column("# Sentences")
        table.add_row("Train", str(len(self.train_data)))
        table.add_row("Validation", str(len(self.dev_data)))
        table.add_row("Test", str(len(self.test_data)))
        console.print(f"Dataset name: [bold magenta]{self.dataset_name}[/bold magenta]")
        console.print(table)

    def collate_fn(self, batch: Dict, tokenizer: tre.Tokenizer) -> ModelInputs:
        raise NotImplementedError


class CoNLL2003NERDataset(Dataset):
    def __init__(
        self, dataset_name: str = "conll2003", max_length: int = 128, **kwargs
    ):
        super().__init__(dataset_name, max_length)
        dataset = load_dataset(self.dataset_name)
        # build labels
        self.labels = Labels()
        self.labels.add_labels(
            {
                n: i
                for i, n in enumerate(
                    dataset["train"].features["ner_tags"].feature.names
                )
            }
        )
        # split data
        self.train_data = dataset["train"]
        self.dev_data = dataset["validation"]
        self.test_data = dataset["test"]

    def collate_fn(self, batch: Dict, tokenizer: tre.Tokenizer) -> ModelInputs:
        batch_out = tokenizer(
            [b["tokens"][:self.max_length] for b in batch],
            return_tensors=True,
            padding=True,
            is_split_into_words=True,
        )
        # prepare for possible label
        # if no labels, prediction batch
        if "ner_tags" in batch[0].keys():
            labels = [[0] + b["ner_tags"][:self.max_length] + [0] for b in batch]
            labels = pad_sequence(
                [torch.as_tensor(sample) for sample in labels],
                batch_first=True,
                padding_value=-100,
            )
            batch_out.update({"labels": torch.as_tensor(labels)})
        return batch_out


class CoNLL2012NERDataset(Dataset):
    def __init__(
        self,
        dataset_name: str = "conll2012_ontonotesv5",
        max_length: int = 128,
        version: str = "english_v12",
        **kwargs,
    ):
        super().__init__(dataset_name, max_length)
        dataset = load_dataset(self.dataset_name, version)
        # build labels
        self.labels = Labels()
        self.labels.add_labels(
            {
                n: i
                for i, n in enumerate(
                    dataset["train"]
                    .features["sentences"][0]["named_entities"]
                    .feature.names
                )
            }
        )
        # split data
        self.train_data = [
            sentence
            for sentences in dataset["train"]["sentences"]
            for sentence in sentences
        ]
        self.dev_data = [
            sentence
            for sentences in dataset["validation"]["sentences"]
            for sentence in sentences
        ]
        self.test_data = [
            sentence
            for sentences in dataset["test"]["sentences"]
            for sentence in sentences
        ]

    def collate_fn(self, batch: Dict, tokenizer: tre.Tokenizer) -> ModelInputs:
        batch_out = tokenizer(
            [b["words"][:self.max_length] for b in batch],
            return_tensors=True,
            padding=True,
            is_split_into_words=True,
        )
        # prepare for possible label
        # if no labels, prediction batch
        if "named_entities" in batch[0].keys():
            labels = [[0] + b["named_entities"][:self.max_length] + [0] for b in batch]
            labels = pad_sequence(
                [torch.as_tensor(sample) for sample in labels],
                batch_first=True,
                padding_value=-100,
            )
            batch_out.update({"labels": torch.as_tensor(labels)})
        return batch_out
