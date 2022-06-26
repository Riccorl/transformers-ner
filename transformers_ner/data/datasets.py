from typing import Dict

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
    def __init__(self):
        self.dataset_name = "dataset"
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


class CoNLL2003NERDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.dataset_name = "conll2003"
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

    @staticmethod
    def collate_fn(batch: Dict, tokenizer: tre.Tokenizer) -> ModelInputs:
        batch_out = tokenizer(
            [b["tokens"] for b in batch],
            return_tensors=True,
            padding=True,
            is_split_into_words=True,
        )
        # prepare for possible label
        # if no labels, prediction batch
        if "ner_tags" in batch[0].keys():
            labels = [[0] + b["ner_tags"] + [0] for b in batch]
            labels = pad_sequence(
                [torch.as_tensor(sample) for sample in labels],
                batch_first=True,
                padding_value=-100,
            )
            batch_out.update({"labels": torch.as_tensor(labels)})
        return batch_out


class CoNLL2012NERDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.dataset_name = "conll2012_ontonotesv5"
        dataset = load_dataset(self.dataset_name, "english_v12")
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

    @staticmethod
    def collate_fn(batch: Dict, tokenizer: tre.Tokenizer) -> ModelInputs:
        batch_out = tokenizer(
            [b["words"] for b in batch],
            return_tensors=True,
            padding=True,
            is_split_into_words=True,
        )
        # prepare for possible label
        # if no labels, prediction batch
        if "named_entities" in batch[0].keys():
            labels = [[0] + b["named_entities"] + [0] for b in batch]
            labels = pad_sequence(
                [torch.as_tensor(sample) for sample in labels],
                batch_first=True,
                padding_value=-100,
            )
            batch_out.update({"labels": torch.as_tensor(labels)})
        return batch_out
