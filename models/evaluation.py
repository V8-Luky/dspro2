"""
This module implements the evaluation logic for our ASL models.
"""

import torch
import torch.nn as nn
import lightning as L
import wandb

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from lightning.pytorch.loggers.wandb import WandbLogger
import matplotlib.pyplot as plt

from .asl_model import ASLModel

LABELS = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "Nothing",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "Space",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z"
]


def show_confusion_matrix(targets: torch.Tensor, predictions: torch.Tensor, title: str, color_map: str = "Blues", display_labels: list[str] = LABELS):
    """
    Displays a confusion matrix for the given targets and predictions.
    """
    cm = confusion_matrix(y_true=targets, y_pred=predictions)

    fig, ax = plt.subplots(figsize=(10, 10))

    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    display.plot(cmap=color_map, ax=ax, xticks_rotation="vertical")
    display.ax_.set_title(title)

    return fig, ax


class Evaluation:
    """
    Responsible for running and logging the evaluation for a given model architecture and datamodule.
    """

    def __init__(
        self,
        name: str,
        project: str,
        entity: str,
        model_architecture: nn.Module,
        artifact: str,
        datamodule: L.LightningDataModule,
    ):
        self.name = name
        self.project = project
        self.entity = entity
        self.artifact = artifact
        self.datamodule = datamodule
        self.model_architecture = model_architecture

    def get_model(self):
        run = wandb.init(name=self.name, project=self.project, entity=self.entity)
        artifact = run.use_artifact(self.artifact, type="model")
        artifact_dir = artifact.download()

        checkpoint = torch.load(artifact_dir + "/model.ckpt")
        model = ASLModel(self.model_architecture, nn.CrossEntropyLoss(), torch.optim.Adam(self.model_architecture.parameters()))

        model.load_state_dict(checkpoint["state_dict"])

        return model

    def __call__(self):
        model = self.get_model()

        logger = WandbLogger(name=self.name)

        trainer = L.Trainer(
            accelerator="auto",
            devices="auto",
            logger=logger,
            log_every_n_steps=1,
        )

        trainer.test(model, self.datamodule)
        results = trainer.predict(model, self.datamodule)

        preds, targets = [], []

        for pred, target in results:
            preds.append(pred)
            targets.append(target)

        wandb.finish()
        return torch.cat(preds), torch.cat(targets)
