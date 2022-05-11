import json
from pathlib import Path
from typing import Union, List, Dict, Any

import hydra.utils
from torch.utils.data import Dataset


class MultiCoNERDataset(Dataset):
    def __init__(
        self,
        train_paths: Union[str, Path, List[str], List[Path]],
        dev_paths: Union[str, Path, List[str], List[Path]],
        test_paths: Union[str, Path, List[str], List[Path]],
        name: str,
    ):
        super().__init__()
        self.name = name
        # get root path
        root_path = Path(hydra.utils.to_absolute_path("."))
        # normalize paths
        if isinstance(train_paths, (str, Path)):
            train_paths = [Path(train_paths)]
        if isinstance(dev_paths, (str, Path)):
            dev_paths = [Path(dev_paths)]
        if isinstance(test_paths, (str, Path)):
            test_paths = [Path(test_paths)]
        # add root path to relative paths
        self.train_paths = [root_path / path for path in train_paths]
        self.dev_paths = [root_path / path for path in dev_paths]
        self.test_paths = [root_path / path for path in test_paths]
        self.datasets = {
            "train": self.load_sentences(self.train_paths),
            "validation": self.load_sentences(self.dev_paths),
            "test": self.load_sentences(self.test_paths),
        }

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]

    def load_sentences(self, paths: Union[str, Path, List[str], List[Path]]) -> List[Dict]:
        sentences = []

        if isinstance(paths, (str, Path)):
            paths = [Path(paths)]

        for path in paths:
            with open(path) as f:

                for line in f:

                    line = line.strip()

                    if line.startswith("#\t"):
                        tokens = []
                        labels = []
                    elif line == "":
                        sentences.append({"tokens": tokens, "ner_tags": labels})
                    else:
                        token, label = line.split("\t")
                        tokens.append(token)
                        labels.append(label)

        return sentences
