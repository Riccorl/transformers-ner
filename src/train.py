import omegaconf
import hydra

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.pl_data_modules import NERDataModule
from src.pl_modules import NERModule


def train(conf: omegaconf.DictConfig) -> None:

    # reproducibility
    pl.seed_everything(conf.seed)

    # data module declaration
    pl_data_module = NERDataModule(conf)

    # main module declaration
    pl_module = NERModule(conf)

    # callbacks declaration
    callbacks_store = []

    if conf.apply_early_stopping:
        callbacks_store.append(
            EarlyStopping(
                monitor=conf.monitor_var,
                mode=conf.monitor_var_mode,
                patience=conf.patience,
            )
        )

    callbacks_store.append(
        ModelCheckpoint(
            monitor=conf.monitor_var,
            dirpath="checkpoints",
            save_top_k=conf.save_top_k,
            verbose=True,
            mode=conf.monitor_var_mode,
            filename="{epoch}.{val_loss:.2f}",
        )
    )

    # trainer
    trainer = pl.Trainer(
        gpus=conf.gpus,
        accumulate_grad_batches=conf.gradient_acc_steps,
        gradient_clip_val=conf.gradient_clip_value,
        val_check_interval=conf.val_check_interval,
        callbacks=callbacks_store,
        max_epochs=conf.max_epochs,
        precision=conf.precision,
    )

    # module fit
    trainer.fit(pl_module, datamodule=pl_data_module)


@hydra.main(config_path="../conf", config_name="root")
def main(conf: omegaconf.DictConfig):
    train(conf)


if __name__ == "__main__":
    main()
