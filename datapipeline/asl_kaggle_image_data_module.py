import os

import kagglehub

import lightning as L
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import Dataset, DataLoader

from asl_image_data_module import ASLImageDataModule


class ASLKaggleImageDataModule(ASLImageDataModule):
    DATASET_NAME = "kapillondhe/american-sign-language"
    DATASET_ROOT = "ASL_Dataset"
    CACHE_ENVIRON = "KAGGLEHUB_CACHE"

    def __init__(self, path: str, train_split_folder: str = "Train", val_split_folder: str = "Valid", test_split_folder: str = "Test", batch_size: int = 32, num_workers: int = 64):
        if path.endswith(ASLKaggleImageDataModule.DATASET_ROOT):
            path = "/".join(path.split("/")[:-1])

        self.download_path = path

        super().__init__(path=f"{path}/{ASLKaggleImageDataModule.DATASET_ROOT}", train_split_folder=train_split_folder,
                         val_split_folder=val_split_folder, test_split_folder=test_split_folder, batch_size=batch_size, num_workers=num_workers)

    def prepare_data(self):
        previous_path = os.environ[ASLKaggleImageDataModule.CACHE_ENVIRON]
        if self.path:
            os.environ[ASLKaggleImageDataModule.CACHE_ENVIRON] = self.download_path

        path = kagglehub.dataset_download(ASLKaggleImageDataModule.DATASET_NAME)

        # TODO: Do split as Luca defined it

        os.environ[ASLKaggleImageDataModule.CACHE_ENVIRON] = previous_path
