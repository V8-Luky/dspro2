import torch.nn as nn
import lightning as L
import wandb


from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from typing import Callable

from .sweep_helper import *
from .asl_model import ASLModel


ENTITY_NAME = "dspro2-silent-speech"
PROJECT_NAME = "silent-speech"

MAX_EPOCHS = 200

run_id = 0


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


def train_model(name_prefix: str, get_model: Callable[[dict], nn.Module], datamodule: L.LightningDataModule, get_optimizer: Callable[[dict, nn.Module], torch.optim.Optimizer] = get_optimizer, seed: int = 42):
    global run_id
    run_id += 1

    L.seed_everything(seed)

    wandb.init(name=f"{name_prefix}-{run_id}")

    wandb_logger = WandbLogger(log_model=True)

    config = wandb.config
    model = get_model(config)

    optimizer_params = config[OPTIMIZER]
    optimizer = get_optimizer(optimizer_params, model)

    learning_rate_scheduler_params = config[LEARNING_RATE_SCHEDULER]
    scheduler = get_learning_rate_scheduler(learning_rate_scheduler_params, optimizer)

    asl_model = ASLModel(model=model, criterion=nn.CrossEntropyLoss(), optimizer=optimizer, lr_scheduler=scheduler)

    train(
        model=asl_model,
        datamodule=datamodule,
        logger=wandb_logger,
        seed=seed
    )

    wandb.finish()


def get_default_callbacks():
    return [
        LearningRateMonitor(logging_interval="step", log_momentum=True, log_weight_decay=True),
        ModelCheckpoint(monitor=ASLModel.VALID_ACCURACY, filename="{epoch:02d}-{valid_accuracy:.2f}", save_top_k=3, mode="max"),
        ModelCheckpoint(monitor="epoch", filename="latest-{epoch:02d}", save_top_k=1, mode="max"),
        EarlyStopping(monitor=ASLModel.VALID_ACCURACY, patience=5, verbose=True, mode="max"),
        EarlyStopping(monitor=ASLModel.TRAIN_ACCURACY, patience=5, verbose=True, mode="max"),
    ]


def sweep(sweep_config: dict, count: int, training_procedure):
    """Starts a W&B sweep with the given configuration and count.

    Args:
        sweep_config (dict): The configuration for the sweep.
        count (int): The number of runs to execute.
        training_procedure (function): The function to run for each sweep run. Should setup the model for the training based on the sweep configuration accessible by wandb.config.
    """

    global run_id
    run_id = 0

    sweep_id = wandb.sweep(sweep=sweep_config, project=PROJECT_NAME, entity=ENTITY_NAME)
    wandb.agent(sweep_id=sweep_id, function=training_procedure, count=count)
    wandb.api.stop_sweep(sweep_id)
    wandb.teardown()
