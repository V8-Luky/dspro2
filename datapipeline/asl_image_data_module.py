import fiftyone.zoo as foz
import lightning as L
import torch
import torchvision.datasets as datasets
import torchvision.transforms.v2 as transforms

from torch.utils.data import Dataset, DataLoader

from .asl_transforms import ExtractHand, RandomBackgroundNoise, RandomRealLifeBackground


class DefaultImageNetTransforms():
    SPLIT_TRAIN = "train"
    SPLIT_VALID = "validation"
    SPLIT_TEST = "test"

    def __init__(self):
        self.TRAIN_W_NOISE = self.get_transforms([0.8, 0.2], background_split=self.SPLIT_TRAIN)
        self.VALID_W_NOISE = self.get_transforms([0.5, 0.5], background_split=self.SPLIT_VALID)
        self.TEST_W_NOISE = self.get_transforms([0.2, 0.8], background_split=self.SPLIT_TEST)
        self.TRAIN = self.get_transforms_real_life_backgrounds_only(background_split=self.SPLIT_TRAIN)
        self.VALID = self.get_transforms_real_life_backgrounds_only(background_split=self.SPLIT_VALID)
        self.TEST = self.get_transforms_real_life_backgrounds_only(background_split=self.SPLIT_TEST)

    def get_background_images(self, split: str = SPLIT_TRAIN):
        """Gets some images that can be used as backgrounds for the ASL dataset."""

        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split=split,
            label_types=["classifications"],
            max_samples=1000,
        )
        return [sample.filepath for sample in dataset]

    def get_transforms_real_life_backgrounds_only(self, background_split: str = SPLIT_TRAIN):
        return self.get_transforms([0.0, 1.0], background_split=background_split)

    def get_transforms(self, noise_vs_background_probability: list[float], background_split: str = SPLIT_TRAIN):
        assert len(noise_vs_background_probability) == 2, "noise_vs_background_probability should be a list of two floats"

        img_size = 224

        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            ExtractHand(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.75, contrast=0.3, saturation=0.75, hue=0.35),
            transforms.RandomAffine(degrees=5, shear=10, scale=(0.8, 1.2), translate=(0.1, 0.1)),
            transforms.RandomPerspective(distortion_scale=0.25, p=0.5),
            transforms.RandomChoice([RandomBackgroundNoise(), RandomRealLifeBackground(
                backgrounds=self.get_background_images(split=background_split))], p=noise_vs_background_probability),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet stats
        ])


DEFAULT_TRANSFORMS = DefaultImageNetTransforms()


class ASLImageDataModule(L.LightningDataModule):
    def __init__(self, path: str, train_transforms=DEFAULT_TRANSFORMS.TRAIN_W_NOISE, valid_transforms=DEFAULT_TRANSFORMS.VALID_W_NOISE, test_transforms=DEFAULT_TRANSFORMS.TEST_W_NOISE, train_split_folder: str = "Train", val_split_folder: str = "Valid", test_split_folder: str = "Test", batch_size: int = 32, num_workers: int = 64):
        super().__init__()
        self.path = path
        self.train_split_folder = train_split_folder
        self.valid_split_folder = val_split_folder
        self.test_split_folder = test_split_folder
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transforms = train_transforms
        self.valid_transforms = valid_transforms
        self.test_transforms = test_transforms

    def setup(self, stage: str):
        if stage == "fit" or stage is None:
            self.train_dataset = datasets.ImageFolder(root=f"{self.path}/{self.train_split_folder}", transform=self.train_transforms)
            self.valid_dataset = datasets.ImageFolder(root=f"{self.path}/{self.valid_split_folder}", transform=self.valid_transforms)
        if stage == "test" or stage is None:
            self.test_dataset = datasets.ImageFolder(root=f"{self.path}/{self.test_split_folder}", transform=self.test_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)
