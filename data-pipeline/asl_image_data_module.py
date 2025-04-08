import lightning as L
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import Dataset, DataLoader

class ASLImageDataModule(L.LightningDataModule):
    def __init__(self, path: str, train_split_folder: str = "Train", val_split_folder: str = "Valid", test_split_folder: str = "Test", batch_size: int = 32):
        super().__init__()
        self.path = path
        self.train_split_folder = train_split_folder
        self.valid_split_folder = val_split_folder
        self.test_split_folder = test_split_folder
        self.batch_size = batch_size

        self.default_transforms =  transforms.Compose([
            # transforms.Resize((img_size, img_size)),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(10),
            # transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            # transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def prepare_data(self):
        # Download the dataset if needed
        return super().prepare_data()

    def setup(self, stage: str):
        if stage == "fit" or stage is None:
            self.train_dataset = datasets.ImageFolder(root=f"{self.path}/{self.train_split_folder}", transform=self.default_transforms)
            self.valid_dataset = datasets.ImageFolder(root=f"{self.path}/{self.valid_split_folder}", transform=self.default_transforms)
        if stage == "test" or stage is None:
            self.test_dataset = datasets.ImageFolder(root=f"{self.path}/{self.test_split_folder}", transform=self.default_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
