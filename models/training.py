import lightning as L
import wandb

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

from .asl_model import ASLModel

ENTITY_NAME = "dspro2-silent-speech"
PROJECT_NAME = "silent-speech"

MAX_EPOCHS = 200


def train(model: ASLModel, datamodule: L.LightningDataModule, logger: WandbLogger, callbacks: list[L.Callback] = None, seed: int = 42):
    """Trains the model using the given datamodule and logger.
    Args:
        model (ASLModel): The model to train.
        datamodule (L.LightningDataModule): The datamodule containing the training and validation data.
        logger (WandbLogger): The logger to use for logging metrics.
        callbacks (list[L.Callback], optional): List of callbacks to use during training. Defaults to None.
        seed (int, optional): Seed for reproducibility. Defaults to 42.
    """

    L.seed_everything(seed)

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
    wandb.finish()


def get_default_callbacks():
    return [
        LearningRateMonitor(logging_interval="step", log_momentum=True, log_weight_decay=True),
        ModelCheckpoint(monitor=ASLModel.VALID_ACCURACY, filename="{epoch:02d}-{valid_accuracy:.2f}", save_top_k=3, mode="max"),
        ModelCheckpoint(moonitor="epoch", filename="latest-{epoch:02d}", save_top_k=1, mode="max"),
        EarlyStopping(monitor=ASLModel.VALID_ACCURACY, patience=5, verbose=True, mode="max"),
        EarlyStopping(monitor=ASLModel.TRAIN_ACCURACY, patience=5, verbose=True, mode="max"),
    ]


def sweep(sweep_config: dict, count: int, training_procedure):
    """Starts a W&B sweep with the given configuration and count.

    Args:
        sweep_config (dict): The configuration for the sweep.
        count (int): The number of runs to execute.
        training_procedure (function): The function to run for each sweep run. Should setup the model for the training based on the sweep configuratio accessible by wandb.config.
    """

    sweep_id = wandb.sweep(sweep=sweep_config, project=PROJECT_NAME, entity=ENTITY_NAME)
    wandb.agent(sweep_id=sweep_id, function=training_procedure, count=count)
    wandb.teardown()
