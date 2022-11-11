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

from selfDefineDataset import WeatherDataset

import warnings

from SetModel import model
from labelProcessing import period_dict, weather_dict

warnings.filterwarnings("ignore")

import glob
test_df = pd.DataFrame({'filename': glob.glob('./test_images/*.jpg')})
test_df['period'] = 0
test_df['weather'] = 0
test_df = test_df.sort_values(by='filename')


test_dataset = WeatherDataset(test_df)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
model.eval()
period_pred = []
weather_pred = []
for i, (x, y1, y2) in enumerate(test_loader):
    pred1, pred2 = model(x)
    period_pred += period_dict[pred1.argmax(1).numpy()].tolist()
    weather_pred += weather_dict[pred2.argmax(1).numpy()].tolist()

test_df['period'] = period_pred
test_df['weather'] = weather_pred
submit_json = {
    'annotations':[]
}

for row in test_df.iterrows():
    submit_json['annotations'].append({
        'filename':  row[1].filename.split('/')[-1],
        'period': row[1].period,
        'weather': row[1].weather,
    })

with open('submit.json', 'w') as up:
    json.dump(submit_json, up)