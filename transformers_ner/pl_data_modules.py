from typing import Any, Union, List, Optional

import pytorch_lightning as pl
import torch
import transformers_embedder as tre
from datasets import load_dataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader


class NERDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: str,
        language_model_name: str,
        batch_sizes: DictConfig,
        num_workers: DictConfig,
        *args,
        **kwargs
    ):
        super().__init__()
        # data
        self.train_data = None
        self.dev_data = None
        self.test_data = None
        self.label_dict = None
        self.label_dict_inverted = None
        # params
        self.dataset = dataset
        self.language_model_name = language_model_name
        self.batch_sizes = batch_sizes
        self.num_workers = num_workers
        # tokenizer
        self.tokenizer = tre.Tokenizer(language_model_name)

    def prepare_data(self, *args, **kwargs):
        datasets = load_dataset(self.dataset)
        self.label_dict = {n: i for i, n in enumerate(datasets["train"].features["ner_tags"].feature.names)}
        self.label_dict_inverted = {
            i: n for i, n in enumerate(datasets["train"].features["ner_tags"].feature.names)
        }
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

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        return tuple(b.to(device) for b in batch)

    def collate_fn(self, batch):
        batch_x = self.tokenizer(
            [b["tokens"] for b in batch],
            return_tensors=True,
            padding=True,
        )
        # prepare for possible label
        batch_out = [batch_x]
        # if no labels, prediction batch
        if "ner_tags" in batch[0].keys():
            batch_y = [[0] + b["ner_tags"] + [0] for b in batch]
            batch_y = [self.tokenizer.pad_sequence(y, -100, "word") for y in batch_y]
            batch_y = torch.as_tensor(batch_y)
            batch_out.append(batch_y)
        return tuple(batch_out)
