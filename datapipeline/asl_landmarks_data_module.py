import lightning as L
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import numpy as np



class ASLLandmarksDataModule(L.LightningDataModule):
    def __init__(self, path: str, train_split_folder: str = "Train", val_split_folder: str = "Valid", test_split_folder: str = "Test", batch_size: int = 32):
        super().__init__()
        self.path = path
        self.train_split_folder = train_split_folder
        self.valid_split_folder = val_split_folder
        self.test_split_folder = test_split_folder
        self.batch_size = batch_size

    def setup(self, stage: str):
       
        if stage == "fit" or stage is None:
            self.train_dataset = self.load_asl_dataset(self.path, self.train_split_folder)
            self.valid_dataset = self.load_asl_dataset(self.path, self.valid_split_folder)

        if stage == "test" or stage is None:
            self.test_dataset = self.load_asl_dataset(self.path, self.test_split_folder)
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False)


    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        
    def load_asl_dataset(self, path, split):
        """Loads the numpy files.
        Normalizes the landmarks. 
        Encodes the labels.
        Returns the dataset."""
        device = torch.device("cuda")
        
        X = np.load(f'{path}/{split}/X_landmarks.npy')
        y = np.load(f'{path}/{split}/y_labels.npy')
        y = np.array([label.upper() for label in y])
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        
        X_reshaped = X.reshape(-1, 3)
        X_centered = X_reshaped - X_reshaped.mean(axis=0)
        max_distance = np.max(np.linalg.norm(X_centered, axis=1))
        
        X_normalized = (X_centered / max_distance).reshape(X.shape)
        
        # TODO MODIFY THE VIEW ACCORDING TO THE SHAPE YOU NEED FOR YOUR NN
        X_tensor = torch.tensor(X_normalized, dtype=torch.float32).view(-1, 1, 21, 3).to(device) 
        y_tensor = torch.tensor(y, dtype=torch.long).to(device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        return dataset