from typing import Any, Union, List, Optional

import pytorch_lightning as pl
import torch
import transformer_embedder as tre
from datasets import load_dataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader


class NERDataModule(pl.LightningDataModule):
    """
    FROM LIGHTNING DOCUMENTATION

    A DataModule standardizes the training, val, test splits, data preparation and transforms.
    The main advantage is consistent data splits, data preparation and transforms across models.

    A DataModule implements 5 key methods:

    * **prepare_data** (things to do on 1 GPU/TPU not on every GPU/TPU in distributed mode).
    * **setup**  (things to do on every accelerator in distributed mode).
    * **train_dataloader** the training dataloader.
    * **val_dataloader** the val dataloader(s).
    * **test_dataloader** the test dataloader(s).


    This allows you to share a full dataset without explaining how to download,
    split transform and process the data

    """

    def __init__(self, conf: DictConfig):
        super().__init__()
        self.conf = conf
        self.tokenizer = tre.Tokenizer(self.conf.language_model_name)
        self.label_dict = None
        self.label_dict_inverted = None
        self.train_data = None
        self.dev_data = None
        self.test_data = None

    def prepare_data(self, *args, **kwargs):
        datasets = load_dataset("conll2003")
        self.label_dict = {
            n: i
            for i, n in enumerate(
                datasets["train"].features[f"{self.conf.task}_tags"].feature.names
            )
        }
        self.label_dict_inverted = {
            i: n
            for i, n in enumerate(
                datasets["train"].features[f"{self.conf.task}_tags"].feature.names
            )
        }
        # train
        self.train_data = datasets["train"]
        # dev
        self.dev_data = datasets["validation"]
        # test
        self.test_data = datasets["test"]

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.train_data,
            batch_size=self.conf.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.conf.num_workers,
            shuffle=True,
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.dev_data,
            batch_size=self.conf.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.conf.num_workers,
        )

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_data,
            batch_size=self.conf.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.conf.num_workers,
        )

    def transfer_batch_to_device(
        self, batch: Any, device: Optional[torch.device] = None
    ) -> Any:
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
        if f"{self.conf.task}_tags" in batch[0].keys():
            batch_y = [[0] + b[f"{self.conf.task}_tags"] + [0] for b in batch]
            batch_y = [self.tokenizer.pad_sequence(y, -100, "word") for y in batch_y]
            batch_y = torch.as_tensor(batch_y)
            batch_out.append(batch_y)
        return tuple(batch_out)
