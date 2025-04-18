import os

import shutil
import random

import kagglehub

import lightning as L
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import Dataset, DataLoader

from .asl_image_data_module import ASLImageDataModule, DEFAULT_TRANSFORMS


class ASLKaggleImageDataModule(ASLImageDataModule):
    DATASET_NAME = "kapillondhe/american-sign-language"
    DATASET_ROOT = "ASL_Dataset"
    CACHE_ENVIRON = "KAGGLEHUB_CACHE"

    def __init__(self, path: str, train_transforms=DEFAULT_TRANSFORMS.TRAIN, valid_transforms=DEFAULT_TRANSFORMS.VALID, test_transforms=DEFAULT_TRANSFORMS.TEST, train_split_folder: str = "Train", val_split_folder: str = "Valid", test_split_folder: str = "Test", batch_size: int = 32, num_workers: int = 64):
        if path.endswith(ASLKaggleImageDataModule.DATASET_ROOT):
            path = "/".join(path.split("/")[:-1])

        self.download_path = path

        super().__init__(path=f"{path}/{ASLKaggleImageDataModule.DATASET_ROOT}", train_transforms=train_transforms, valid_transforms=valid_transforms, test_transforms=test_transforms, train_split_folder=train_split_folder,
                         val_split_folder=val_split_folder, test_split_folder=test_split_folder, batch_size=batch_size, num_workers=num_workers)

    def prepare_data(self):
        previous_path = os.environ.get(ASLKaggleImageDataModule.CACHE_ENVIRON, "")
        if self.path:
            os.environ[ASLKaggleImageDataModule.CACHE_ENVIRON] = self.download_path

        path = kagglehub.dataset_download(ASLKaggleImageDataModule.DATASET_NAME)
        path = os.path.join(path, ASLKaggleImageDataModule.DATASET_ROOT)
        self.path = path

        split_dirs = [os.path.join(self.path, self.train_split_folder), os.path.join(
            self.path, self.valid_split_folder), os.path.join(self.path, self.test_split_folder)]
        if all([os.path.exists(split)for split in split_dirs]):
            print("Split folders already exist, skipping distribution.")
            os.environ[ASLKaggleImageDataModule.CACHE_ENVIRON] = previous_path
            return

        self.distribute_files([os.path.join(path, "Train"), os.path.join(path, "Test")], split_dirs, split=(0.6, 0.2, 0.2))

        os.environ[ASLKaggleImageDataModule.CACHE_ENVIRON] = previous_path

    def distribute_files(self, source_folders, target_folders, split=(0.6, 0.2, 0.2)):
        """
        Distributes files from source_folders into target_folders based on the given split.

        Parameters:
            source_folders (list): List of two folder paths containing source files.
            target_folders (list): List of three folder paths to distribute files into.
            split (tuple): A tuple with three values representing the percentage split (default is (0.6, 0.2, 0.2)).

        Returns:
            None
        """
        assert len(source_folders) == 2, "Exactly two source folders required"
        assert len(target_folders) == 3, "Exactly three target folders required"

        for class_name in os.listdir(source_folders[0]):
            # Skip if not a directory
            if not os.path.isdir(os.path.join(source_folders[0], class_name)):
                continue

            # Collect files from both source folders for the current class
            all_files = []
            for src in source_folders:
                class_dir = os.path.join(src, class_name)
                if not os.path.isdir(class_dir):
                    continue
                files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
                all_files.extend(files)

            # Shuffle to randomize
            # random.shuffle(all_files)

            total = len(all_files)
            split1 = int(split[0] * total)
            split2 = split1 + int(split[1] * total)

            split_sets = [
                all_files[:split1],     # Train
                all_files[split1:split2],  # Val
                all_files[split2:]        # Test
            ]

            # Create class subfolders in each target split directory
            for i, subset in enumerate(split_sets):
                split_folder = os.path.join(target_folders[i], class_name)
                os.makedirs(split_folder, exist_ok=True)
                for file_path in subset:
                    shutil.move(file_path, split_folder)

            print(f"Class '{class_name}': {len(split_sets[0])} train, {len(split_sets[1])} val, {len(split_sets[2])} test")
