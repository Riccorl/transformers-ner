from typing import Any, Union, List, Optional

import hydra
import pytorch_lightning as pl
import torch
import transformers_embedder as tre
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from data.datasets import Dataset
from data.labels import Labels


class NERDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: Union[DictConfig, Dataset],
        language_model_name: str,
        batch_sizes: DictConfig,
        num_workers: DictConfig,
        labels: Optional[Labels] = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        # data
        self.dataset: Union[DictConfig, Dataset] = dataset
        self.labels = labels
        # params
        self.language_model_name = language_model_name
        self.batch_sizes = batch_sizes
        self.num_workers = num_workers
        # tokenizer
        self.tokenizer = tre.Tokenizer(language_model_name, *args, **kwargs)

    def prepare_data(self, *args, **kwargs):
        if isinstance(self.dataset, DictConfig):
            self.dataset = hydra.utils.instantiate(self.dataset)
        if self.labels is None:
            self.labels = self.dataset.labels
        # print some stats
        self.dataset.print_stats()

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.dataset.train_data,
            batch_size=self.batch_sizes.train,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers.train,
            shuffle=True,
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.dataset.dev_data,
            batch_size=self.batch_sizes.dev,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers.dev,
        )

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.dataset.test_data,
            batch_size=self.batch_sizes.test,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers.test,
        )

    def transfer_batch_to_device(
        self, batch: Any, device: torch.device, dataloader_idx: int
    ) -> Any:
        return batch.to(device)

    def collate_fn(self, batch: Any) -> Any:
        return self.dataset.collate_fn(batch, self.tokenizer)
