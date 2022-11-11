import io
import math, json
import numpy as np
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt
import paddle
import paddle.nn.functional as F
import paddle.vision.transforms as T
from paddle.io import DataLoader, Dataset

import warnings

from dataReading import train_json

warnings.filterwarnings("ignore")


class WeatherDataset(Dataset):
    def __init__(self, df):
        super(WeatherDataset, self).__init__()
        self.df = df

        self.transform = T.Compose([
            T.Resize(size=(340, 340)),
            T.RandomCrop(size=(256, 256)),
            T.RandomRotation(10),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ToTensor(),
            T.Normalize(mean=0.5, std=0.5)
        ])

    def __getitem__(self, index):
        file_name = self.df['filename'].iloc[index]
        img = Image.open(file_name)
        img = self.transform(img)
        return img, \
               paddle.to_tensor(self.df['period'].iloc[index]), \
               paddle.to_tensor(self.df['weather'].iloc[index])

    def __len__(self):
        return len(self.df)


train_dataset = WeatherDataset(train_json.iloc[:-500])
val_dataset = WeatherDataset(train_json.iloc[-500:])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)