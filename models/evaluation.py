import torch
import torch.nn as nn
import lightning as L
import wandb

from lightning.pytorch.loggers.wandb import WandbLogger

from .asl_model import ASLModel


class Evaluation:
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

        logger = WandbLogger(name=self.name, log_model=True)

        trainer = L.Trainer(
            accelerator="auto",
            devices="auto",
            logger=logger,
            log_every_n_steps=100,
        )

        trainer.test(model, self.datamodule)

        wandb.finish()
