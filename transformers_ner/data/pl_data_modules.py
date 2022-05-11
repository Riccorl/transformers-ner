from typing import Any, Union, List, Optional

import pytorch_lightning as pl
import torch
import transformers_embedder as tre
from datasets import load_dataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import hydra
from torch.nn.utils.rnn import pad_sequence
from transformers_embedder.tokenizer import ModelInputs

from data.labels import Labels


class NERDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: Union[str, DictConfig],
        language_model_name: str,
        batch_sizes: DictConfig,
        num_workers: DictConfig,
        labels: Optional[Labels] = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        # data
        self.train_data = None
        self.dev_data = None
        self.test_data = None
        self.labels = labels
        # params
        self.dataset = dataset
        self.language_model_name = language_model_name
        self.batch_sizes = batch_sizes
        self.num_workers = num_workers
        # tokenizer
        self.tokenizer = tre.Tokenizer(language_model_name)

    def prepare_data(self, *args, **kwargs):
        # load dataset from HF
        if isinstance(self.dataset, str):
            datasets = load_dataset(self.dataset)
        elif isinstance(self.dataset, DictConfig):
            # load datasets from file
            datasets = hydra.utils.instantiate(self.dataset).datasets
        else:
            raise ValueError(f"dataset must be `str` or `DictConfig`, got `{type(self.dataset)}`")
        # build labels
        if not self.labels:
            self.labels = Labels()
            self.labels.add_labels(
                {n: i for i, n in enumerate(datasets["train"].features["ner_tags"].feature.names)}
            )
            # for sample in datasets["train"]:
            #     self.labels.add_labels(sample["ner_tags"])
        # split data
        self.train_data = datasets["train"]
        self.dev_data = datasets["validation"]
        self.test_data = datasets["test"]

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.train_data,
            batch_size=self.batch_sizes.train,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers.train,
            shuffle=True,
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.dev_data,
            batch_size=self.batch_sizes.dev,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers.dev,
        )

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_data,
            batch_size=self.batch_sizes.test,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers.test,
        )

    def transfer_batch_to_device(
        self, batch: Any, device: torch.device, dataloader_idx: int
    ) -> Any:
        return batch.to(device)

    def collate_fn(self, batch):
        batch_out = self.tokenizer(
            [b["tokens"] for b in batch],
            return_tensors=True,
            padding=True,
            is_split_into_words=True,
        )
        # prepare for possible label
        # if no labels, prediction batch
        if "ner_tags" in batch[0].keys():
            labels = [[0] + b["ner_tags"] + [0] for b in batch]
            # labels = [
            #     [self.labels.get_index_from_label(l) for l in b["ner_tags"]]
            #     for b in batch
            # ]
            labels = pad_sequence(
                [torch.as_tensor(sample) for sample in labels],
                batch_first=True,
                padding_value=-100,
            )
            batch_out.update({"labels": torch.as_tensor(labels)})
        return batch_out
