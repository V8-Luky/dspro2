import torch.nn as nn

class ASLClassifierImproved(nn.Module):
    def __init__(self, input_size=63, num_classes=28):
        super(ASLClassifierImproved, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(63, 64),  
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 28)
        )

    def forward(self, x):
        return self.model(x)