import logging

import hydra
import torch
from datasets import load_metric
from omegaconf import omegaconf
from tqdm import tqdm

from pl_modules import NERModule

log = logging.getLogger(__name__)


def compute_metrics(predictions, labels, sentence_length, label_list, metric):
    true_predictions = []
    true_labels = []
    for prediction, label, s in zip(predictions, labels, sentence_length):
        true_predictions.append([label_list[p] for p in prediction[1 : s - 2]])
        true_labels.append([label_list[l] for l in label[1 : s - 2]])
    return metric.compute(predictions=true_predictions, references=true_labels)


def predict(conf: omegaconf.DictConfig):
    # model loading
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Using {} as device".format(device))
    log.info("Loading model")
    model = NERModule.load_from_checkpoint(checkpoint_path=conf.evaluate.checkpoint_path)
    model.to(device)
    model.eval()
    # metric
    metric = load_metric("seqeval")
    # data module
    data_module = hydra.utils.instantiate(conf.data.datamodule)
    data_module.prepare_data()
    # predict
    predictions, labels, sentence_length = [], [], []
    for batch in tqdm(data_module.test_dataloader(), desc="Predictions"):
        x, y = model.transfer_batch_to_device(batch, device)
        y_hat = model(x)
        predictions += torch.argmax(y_hat, dim=-1).tolist()
        labels += y.tolist()
        sentence_length += x["sentence_length"]

    # overall score print
    log.info("Overall scores")
    scores = compute_metrics(predictions, labels, sentence_length, data_module.label_dict_inverted, metric)
    for k, v in scores.items():
        log.info(f"{k}: {v}")


@hydra.main(config_path="../conf", config_name="root")
def main(conf: omegaconf.DictConfig):
    predict(conf)


if __name__ == "__main__":
    main()
