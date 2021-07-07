import json
from pathlib import Path
from typing import Optional

import hydra
import omegaconf
import torch
from onnxruntime.quantization import quantize_dynamic, QuantType
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pl_modules import NERModule


def train(conf: omegaconf.DictConfig) -> None:

    # reproducibility
    pl.seed_everything(conf.train.seed)

    if conf.train.pl_trainer.fast_dev_run:
        hydra.utils.log.info(
            f"Debug mode <{conf.train.pl_trainer.fast_dev_run}>. Forcing debugger configuration"
        )
        # Debuggers don't like GPUs nor multiprocessing
        conf.train.pl_trainer.gpus = 0
        conf.train.pl_trainer.precision = 32
        conf.data.datamodule.num_workers = 0
        # Switch wandb mode to offline to prevent online logging
        conf.logging.wandb_arg.mode = "offline"

    # data module declaration
    hydra.utils.log.info(f"Instantiating the Data Module")
    pl_data_module = hydra.utils.instantiate(conf.data.datamodule)
    pl_data_module.prepare_data()

    # main module declaration
    hydra.utils.log.info(f"Instantiating the Model")
    pl_module: pl.LightningModule = hydra.utils.instantiate(
        conf.model, labels=pl_data_module.label_dict
    )

    # callbacks declaration
    callbacks_store = []

    if conf.train.early_stopping_callback is not None:
        early_stopping_callback: EarlyStopping = hydra.utils.instantiate(
            conf.train.early_stopping_callback
        )
        callbacks_store.append(early_stopping_callback)

    if conf.train.model_checkpoint_callback is not None:
        model_checkpoint_callback: ModelCheckpoint = hydra.utils.instantiate(
            conf.train.model_checkpoint_callback
        )
        callbacks_store.append(model_checkpoint_callback)

    logger: Optional[WandbLogger] = None
    if conf.logging.log:
        hydra.utils.log.info(f"Instantiating Wandb Logger")
        logger = hydra.utils.instantiate(conf.logging.wandb_arg)
        logger.watch(pl_module, **conf.logging.watch)

    # trainer
    hydra.utils.log.info(f"Instantiating the Trainer")
    trainer: Trainer = hydra.utils.instantiate(
        conf.train.pl_trainer, callbacks=callbacks_store, logger=logger
    )
    # module fit
    trainer.fit(pl_module, datamodule=pl_data_module)
    # module test
    trainer.test(pl_module, datamodule=pl_data_module)

    if conf.train.export and not conf.train.pl_trainer.fast_dev_run:
        # export model stuff
        # export best model weights
        export_path = Path(logger.experiment.dir) / "model"
        # export_path = Path(logger.save_dir) / logger.name / logger.version / "files"
        export_path.mkdir(parents=True, exist_ok=True)
        best_model = NERModule.load_from_checkpoint(
            model_checkpoint_callback.best_model_path, labels=pl_data_module.label_dict
        )
        torch.save(
            best_model.state_dict(),
            export_path / "weights.pt",
        )
        # save labels
        json.dump(pl_data_module.label_dict, open(export_path / "labels.json", "w"))
        inputs, _ = next(iter(pl_data_module.train_dataloader()))
        # onnx accepts only Tuples
        onnx_inputs = (
            inputs.input_ids,
            inputs.attention_mask,
            inputs.token_type_ids,
            inputs.offsets,
        )
        # export onnx
        torch.onnx.export(
            best_model,
            onnx_inputs,
            export_path / "weights.onnx",
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=12,  # the ONNX version to export the model to
            do_constant_folding=True,  # whether to execute constant folding for optimization
            use_external_data_format=bool(
                "large" in conf.model.language_model_name
            ),  # export models larger than 2gb
            input_names=[
                "input_ids",
                "attention_mask",
                "token_type_ids",
                "offsets",
            ],  # the model's input names
            output_names=["ner_tags"],  # the model's output names
            verbose=False,
            dynamic_axes={
                "input_ids": {
                    0: "batch_size",
                    1: "batch_length",
                },  # variable length axes
                "offsets": {
                    0: "batch_size",
                    1: "batch_length",
                },  # variable length axes
                "attention_mask": {
                    0: "batch_size",
                    1: "batch_length",
                },  # variable length axes
                "token_type_ids": {
                    0: "batch_size",
                    1: "batch_length",
                },  # variable length axes
                "ner_tags": {
                    0: "batch_size",
                    1: "batch_length",
                },  # variable length axes
            },
        )
        quantize_dynamic(
            model_input=export_path / "weights.onnx",
            model_output=export_path / "weights.quantized.onnx",
            per_channel=True,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QUInt8,
            optimize_model=True,
        )


@hydra.main(config_path="../conf", config_name="root")
def main(conf: omegaconf.DictConfig):
    train(conf)


if __name__ == "__main__":
    main()
