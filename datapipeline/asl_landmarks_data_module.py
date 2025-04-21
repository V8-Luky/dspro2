import lightning as L
import torch

from torch.utils.data import Dataset, DataLoader


class ASLLandmarksDataModule(L.LightningDataModule):
    def __init__(self, path: str, train_split_folder: str = "Train", val_split_folder: str = "Valid", test_split_folder: str = "Test", batch_size: int = 32, num_workers: int = 64):
        super().__init__()
        self.path = path
        self.train_split_folder = train_split_folder
        self.valid_split_folder = val_split_folder
        self.test_split_folder = test_split_folder
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        # TODO: Add logic on how to read numpy files with landmarks here
        # TODO: Read landmark files here and create torch datasets for them.
        # if stage == "fit" or stage is None:
        # self.train_dataset = datasets.ImageFolder(root=f"{self.path}/{self.train_split_folder}", transform=self.default_transforms)
        # self.valid_dataset = datasets.ImageFolder(root=f"{self.path}/{self.valid_split_folder}", transform=self.default_transforms)
        # if stage == "test" or stage is None:
        # self.test_dataset = datasets.ImageFolder(root=f"{self.path}/{self.test_split_folder}", transform=self.default_transforms)
        pass

    def train_dataloader(self):
        # TODO: Return landmark dataloader
        # return DataLoader(self.train_dataset, batch_size=self.batch_size)
        pass

    def val_dataloader(self):
        # TODO: Return landmark dataloader
        # return DataLoader(self.valid_dataset, batch_size=self.batch_size)
        pass

    def test_dataloader(self):
        # TODO: Return landmark dataloader
        # return DataLoader(self.test_dataset, batch_size=self.batch_size)
        pass
