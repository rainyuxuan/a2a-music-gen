import numpy as np

from features.preprocessor import *
import torch
import torch.nn as nn

from trainer import *
# from data.dataset_fix_size import *



class Config:
    input_len = 256
    out_len = 128


class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        input_size = self.config.input_len
        output_size = self.config.out_len
        self.fc1 = nn.Linear(input_size, input_size*2)
        self.fc2 = nn.Linear(input_size*2, input_size*3)
        self.fc3 = nn.Linear(input_size * 3, input_size * 2)
        self.fc4 = nn.Linear(input_size*2, input_size)
        self.fc5 = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        return x

    def generate(self, input: torch.Tensor, new_tokens: int):
        generated_tokens = input.clone()
        for _ in range(new_tokens):
            input = generated_tokens[:, -self.config.input_len:]
            logits = self.forward(input)
            generated_tokens = torch.cat([generated_tokens, logits.unsqueeze(1)], dim=1)
        return generated_tokens


# to load data into input and output format
if __name__ == "__main__":

    model = MLP(Config)
    print(model)
    train_path = "/Users/xunuo/Desktop/a2a-music-gen/data/raw/train_data"
    valid_path = "/Users/xunuo/Desktop/a2a-music-gen/data/raw/test_data"
    train_dataset = WaveDataset(train_path, sr=100)
    valid_dataset = WaveDataset(valid_path, sr=100)
    train_dl = DataLoader(train_dataset, batch_size=3, shuffle=False)
    valid_dl = DataLoader(valid_dataset, batch_size=3, shuffle=False)
    Trainer = Trainer()
    Trainer.set_dataloaders(train_loader=train_dl, val_loader=valid_dl, test_loader=None)
    Trainer.model = model
    Trainer.train(1e-4,1e-4, 10)

