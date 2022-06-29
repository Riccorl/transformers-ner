import importlib.util
from pathlib import Path
from typing import Optional

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar
from pytorch_lightning.loggers import WandbLogger
from rich.console import Console

from data.pl_data_modules import NERDataModule
from models.pl_modules import NERModule


def train(conf: omegaconf.DictConfig) -> None:
    # fancy logger
    console = Console()
    # reproducibility
    pl.seed_everything(conf.train.seed)

    console.log(
        f"Starting training for [bold cyan]{conf.train.model_name}[/bold cyan] model"
    )
    if conf.train.pl_trainer.fast_dev_run:
        console.log(
            f"Debug mode {conf.train.pl_trainer.fast_dev_run}. Forcing debugger configuration"
        )
        # Debuggers don't like GPUs nor multiprocessing
        conf.train.pl_trainer.accelerator = "cpu"
        conf.train.pl_trainer.devices = 1
        conf.train.pl_trainer.strategy = None
        conf.train.pl_trainer.precision = 32
        conf.data.datamodule.num_workers = {
            k: 0 for k in conf.data.datamodule.num_workers
        }
        # Switch wandb to offline mode to prevent online logging
        conf.logging.log = None
        # remove model checkpoint callback
        conf.train.model_checkpoint_callback = None

    # data module declaration
    console.log(f"Instantiating the Data Module")
    pl_data_module: NERDataModule = hydra.utils.instantiate(
        conf.data.datamodule, _recursive_=False
    )
    # force setup to get labels initialized for the model
    pl_data_module.prepare_data()

    # main module declaration
    model_kwargs = {"_recursive_": False, "labels": pl_data_module.labels}
    console.log(f"Instantiating the Model")
    pl_module: NERModule = hydra.utils.instantiate(conf.model, **model_kwargs)

    experiment_logger: Optional[WandbLogger] = None
    experiment_path: Optional[Path] = None
    if conf.logging.log:
        console.log(f"Instantiating Wandb Logger")
        experiment_logger = hydra.utils.instantiate(conf.logging.wandb_arg)
        experiment_logger.watch(pl_module, **conf.logging.watch)

    # callbacks declaration
    callbacks_store = [RichProgressBar()]

    if conf.train.early_stopping_callback is not None:
        early_stopping_callback: EarlyStopping = hydra.utils.instantiate(
            conf.train.early_stopping_callback
        )
        callbacks_store.append(early_stopping_callback)

    model_checkpoint_callback: Optional[ModelCheckpoint] = None
    if conf.train.model_checkpoint_callback is not None:
        model_checkpoint_callback = hydra.utils.instantiate(
            conf.train.model_checkpoint_callback,
            dirpath=experiment_path / "checkpoints" if experiment_path else None,
        )
        callbacks_store.append(model_checkpoint_callback)

    # trainer
    console.log(f"Instantiating the Trainer")
    trainer: Trainer = hydra.utils.instantiate(
        conf.train.pl_trainer, callbacks=callbacks_store, logger=experiment_logger
    )

    model_export: Optional[Path] = None
    if trainer.global_rank == 0:
        if conf.logging.log:
            experiment_path = Path(experiment_logger.experiment.dir)
            # Store the YaML config separately into the wandb dir
            yaml_conf: str = OmegaConf.to_yaml(cfg=conf)
            (experiment_path / "hparams.yaml").write_text(yaml_conf)
            # save labels before starting training
            model_export = experiment_path / "model_export"
            model_export.mkdir(exist_ok=True, parents=True)
            # save labels
            pl_data_module.labels.to_file(model_export / "labels.json")

    # module fit
    trainer.fit(pl_module, datamodule=pl_data_module)

    if trainer.global_rank == 0:
        if model_checkpoint_callback:
            # load best model for testing
            best_pl_module = NERModule.load_from_checkpoint(
                model_checkpoint_callback.best_model_path, labels=pl_data_module.labels
            )
        else:
            best_pl_module = pl_module

        # module test
        trainer.test(best_pl_module, datamodule=pl_data_module)

        if conf.train.export and not conf.train.pl_trainer.fast_dev_run:
            # export model stuff
            best_model = best_pl_module.model
            torch.save(
                best_model.state_dict(),
                model_export / "weights.pt",
            )
            if is_onnx_available():
                from onnxruntime.quantization import quantize_dynamic, QuantType

                inputs = next(iter(pl_data_module.train_dataloader()))
                dynamic_axes = {
                    "input_ids": {
                        0: "batch_size",
                        1: "batch_length",
                    },  # variable length axes
                    "attention_mask": {
                        0: "batch_size",
                        1: "batch_length",
                    },  # variable length axes
                    "offsets": {
                        0: "batch_size",
                        1: "batch_length",
                    },  # variable length axes
                    "ner_tags": {
                        0: "batch_size",
                        1: "batch_length",
                    },  # variable length axes
                }
                # onnx accepts only Tuples
                onnx_inputs = (
                    inputs.input_ids,
                    inputs.attention_mask,
                    inputs.offsets,
                )
                input_names = ["input_ids", "attention_mask", "offsets"]

                # export onnx
                torch.onnx.export(
                    best_model,
                    onnx_inputs,
                    model_export / "weights.onnx",
                    export_params=True,  # store the trained parameter weights inside the model file
                    opset_version=15,  # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names=input_names,  # the model's input names
                    output_names=["ner_tags"],  # the model's output names
                    verbose=False,
                    dynamic_axes=dynamic_axes,
                )
                quantize_dynamic(
                    model_input=model_export / "weights.onnx",
                    model_output=model_export / "weights.quantized.onnx",
                    per_channel=True,
                    activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QUInt8,
                    optimize_model=True,
                )


_onnx_available = importlib.util.find_spec("onnx") is not None


def is_onnx_available():
    return _onnx_available


@hydra.main(config_path="../conf", config_name="default")
def main(conf: omegaconf.DictConfig):
    train(conf)


if __name__ == "__main__":
    main()
