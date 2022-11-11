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
from selfDefineDataset import train_loader,val_loader
import warnings

from SetModel import model

from dataReading import train_json






warnings.filterwarnings("ignore")

optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=0.0001)
criterion = paddle.nn.CrossEntropyLoss()

for epoch in range(0, 40):
    Train_Loss, Val_Loss = [], []
    Train_ACC1, Train_ACC2 = [], []
    Val_ACC1, Val_ACC2 = [], []

    model.train()
    for i, (x, y1, y2) in enumerate(train_loader):
        pred1, pred2 = model(x)
        loss = criterion(pred1, y1) + criterion(pred2, y2)
        Train_Loss.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        Train_ACC1.append((pred1.argmax(1) == y1.flatten()).numpy().mean())
        Train_ACC2.append((pred2.argmax(1) == y2.flatten()).numpy().mean())

    model.eval()
    for i, (x, y1, y2) in enumerate(val_loader):
        pred1, pred2 = model(x)
        loss = criterion(pred1, y1) + criterion(pred2, y2)
        Val_Loss.append(loss.item())
        Val_ACC1.append((pred1.argmax(1) == y1.flatten()).numpy().mean())
        Val_ACC2.append((pred2.argmax(1) == y2.flatten()).numpy().mean())

    if epoch % 1 == 0:
        print(f'\nEpoch: {epoch}')
        print(f'Loss {np.mean(Train_Loss):3.5f}/{np.mean(Val_Loss):3.5f}')
        print(f'period.ACC {np.mean(Train_ACC1):3.5f}/{np.mean(Val_ACC1):3.5f}')
        print(f'weather.ACC {np.mean(Train_ACC2):3.5f}/{np.mean(Val_ACC2):3.5f}')