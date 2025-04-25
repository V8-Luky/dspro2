import lightning as L

import torchmetrics
import torchmetrics.classification

class ASLModel(L.LightningModule):
    TRAIN_ACCURACY = "train_accuracy"
    VALID_ACCURACY = "valid_accuracy"
    TEST_ACCURACY = "test_accuracy"
    TRAIN_LOSS = "train_loss"
    VALID_LOSS = "valid_loss"
    TEST_LOSS = "test_loss"

    def __init__(self, model, criterion, optimizer, lr_scheduler=None):
        super().__init__()

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.train_accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=28, average="micro")
        self.valid_accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=28, average="micro")
        self.test_accuracy = torchmetrics.classification.MulticlassAccuracy(num_classes=28, average="micro")
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        y_hat, loss = self._handle_step(batch, batch_idx, self.train_accuracy, ASLModel.TRAIN_LOSS, ASLModel.TRAIN_ACCURACY)
        return loss

    def validation_step(self, batch, batch_idx):
        self._handle_step(batch, batch_idx, self.valid_accuracy, ASLModel.VALID_LOSS, ASLModel.VALID_ACCURACY)

    def test_step(self, batch, batch_idx):
        self._handle_step(batch, batch_idx, self.test_accuracy, ASLModel.TEST_LOSS, ASLModel.TEST_ACCURACY)

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        y_hat = self(x)
        return y_hat

    def _handle_step(self, batch, batch_idx: int, metric: torchmetrics.Metric, loss_name: str, metric_name: str):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        metric(y_hat, y)

        self.log(loss_name, loss, on_step=True, on_epoch=True)
        self.log(metric_name, metric, on_step=True, on_epoch=True)

        return y_hat, loss

    def configure_optimizers(self):
        if self.lr_scheduler:
            return [self.optimizer], [self.lr_scheduler]
        return self.optimizer
