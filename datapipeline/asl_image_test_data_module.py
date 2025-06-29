import lightning as L
import torch
import torchvision.datasets as datasets
import torchvision.transforms.v2 as transforms

from torch.utils.data import DataLoader

IMG_SIZE = 224


class ASLImageTestDataModule(L.LightningDataModule):
    """
    The datamodule that provides our manual test dataset for the evaluation of our image classification models.
    """

    def __init__(self, path: str, batch_size: int = 32, num_workers: int = 64):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transforms = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def setup(self, stage: str):
        self.dataset = datasets.ImageFolder(root=f"{self.path}", transform=self.transforms)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def predict_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)
