import lightning as L
import wandb

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

from .asl_model import ASLModel

PROJECT_NAME = "silent-speech"
MAX_EPOCHS = 200


def train(run_name: str, model: ASLModel, datamodule: L.LightningDataModule, callbacks: list[L.Callback] = None, seed: int = 42):
    L.seed_everything(seed)

    logger = WandbLogger(name=run_name, project=PROJECT_NAME)

    all_callbacks = get_default_callbacks()
    if callbacks is not None:
        all_callbacks.extend(callbacks)

    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=MAX_EPOCHS,
        logger=logger,
        callbacks=all_callbacks
    )

    trainer.fit(model, datamodule=datamodule)

    for checkpoint in filter(lambda c: isinstance(c, ModelCheckpoint), all_callbacks):
        artifact = wandb.Artifact(name=checkpoint.monitor, description=f"{checkpoint.monitor} checkpoints", type="model")
        artifact.add_dir(checkpoint.dirpath)
        wandb.run.log_artifact(artifact)

    wandb.finish()


def get_default_callbacks():
    return [
        LearningRateMonitor(logging_interval="step", log_momentum=True, log_weight_decay=True),
        ModelCheckpoint(monitor=ASLModel.VALID_ACCURACY, filename="{epoch:02d}-{valid_accuracy:.2f}", save_top_k=3, mode="max"),
        EarlyStopping(monitor=ASLModel.VALID_ACCURACY, patience=5, verbose=True, mode="max"),
        EarlyStopping(monitor=ASLModel.TRAIN_ACCURACY, patience=5, verbose=True, mode="max"),
    ]
