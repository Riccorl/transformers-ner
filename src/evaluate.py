import logging

import hydra
import torch
from omegaconf import omegaconf
from datasets import load_metric
from tqdm import tqdm

from pl_data_modules import NERDataModule
from pl_modules import NERModule

log = logging.getLogger(__name__)


def compute_metrics(predictions, labels, sentence_length, label_list, metric):
    predictions = torch.argmax(predictions, dim=-1)
    predictions = predictions.tolist()
    labels = labels.tolist()

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
    conf.lm_fine_tune = False
    model = NERModule.load_from_checkpoint(checkpoint_path=conf.checkpoint_path)
    model.to(device)
    model.eval()
    # metric
    metric = load_metric("seqeval")
    # data module
    conf.language_model_name = model.hparams.language_model_name
    data_module = NERDataModule(conf)
    data_module.prepare_data()
    # score data structure
    overall_scores = {
        "precision": [],
        "recall": [],
        "f1": [],
        "accuracy": [],
    }
    # predict
    for batch in tqdm(data_module.test_dataloader(), desc="Predictions"):
        x, y = model.transfer_batch_to_device(batch, device)
        y_hat = model(x)
        scores = compute_metrics(
            y_hat, y, x["sentence_length"], data_module.label_dict_inverted, metric
        )
        for k, v in scores.items():
            if k in overall_scores.keys():
                overall_scores[k] += v
    # overall score print
    log.info("Overal scores")
    for k, v in overall_scores.items():
        log.info(f"{k}: {v}")


@hydra.main(config_path="../conf", config_name="root")
def main(conf: omegaconf.DictConfig):
    predict(conf)


if __name__ == "__main__":
    main()
