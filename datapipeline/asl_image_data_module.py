import lightning as L
import torch
import torchvision.datasets as datasets

from torch.utils.data import Dataset, DataLoader


class ASLImageDataModule(L.LightningDataModule):
    def __init__(self, path: str, transforms, train_split_folder: str = "Train", val_split_folder: str = "Valid", test_split_folder: str = "Test", batch_size: int = 32, num_workers: int = 64):
        super().__init__()
        self.path = path
        self.train_split_folder = train_split_folder
        self.valid_split_folder = val_split_folder
        self.test_split_folder = test_split_folder
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transforms = transforms

    def setup(self, stage: str):
        if stage == "fit" or stage is None:
            self.train_dataset = datasets.ImageFolder(root=f"{self.path}/{self.train_split_folder}", transform=self.transforms)
            self.valid_dataset = datasets.ImageFolder(root=f"{self.path}/{self.valid_split_folder}", transform=self.transforms)
        if stage == "test" or stage is None:
            self.test_dataset = datasets.ImageFolder(root=f"{self.path}/{self.test_split_folder}", transform=self.transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
