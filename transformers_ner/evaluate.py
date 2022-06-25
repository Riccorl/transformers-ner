import logging

import hydra
import torch
from omegaconf import omegaconf
from tqdm import tqdm
from utils.scorer import SeqevalScorer

from models.pl_modules import NERModule

log = logging.getLogger(__name__)


def predict(conf: omegaconf.DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model loading
    log.info("Using {} as device".format(device))
    log.info("Loading model")
    model = NERModule.load_from_checkpoint(
        checkpoint_path=conf.evaluate.checkpoint_path
    )
    model.to(device)
    model.eval()

    # data module
    data_module = hydra.utils.instantiate(conf.data.datamodule)
    data_module.prepare_data()

    # metric
    metric = SeqevalScorer()

    # predict
    predictions, labels = [], []
    for batch in tqdm(data_module.test_dataloader(), desc="Predictions"):
        batch = model.transfer_batch_to_device(batch, device)
        predictions_kwargs = {**batch, "compute_predictions": True}
        outputs = model(**predictions_kwargs)
        predictions += outputs["predictions"]
        labels += batch.labels.tolist()

    # overall score print
    log.info("Overall scores")
    scores = metric(predictions, labels)
    for k, v in scores.items():
        log.info(f"{k}: {v}")


@hydra.main(config_path="../conf", config_name="default")
def main(conf: omegaconf.DictConfig):
    predict(conf)


if __name__ == "__main__":
    main()
