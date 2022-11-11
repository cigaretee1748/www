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
warnings.filterwarnings("ignore")

from paddle.vision.models import resnet18

from paddle.vision.models import resnet18

class WeatherModel(paddle.nn.Layer):
    def __init__(self):
        super(WeatherModel, self).__init__()
        backbone = resnet18(pretrained=True)
        backbone.fc = paddle.nn.Identity()
        self.backbone = backbone
        self.fc1 = paddle.nn.Linear(512, 4)
        self.fc2 = paddle.nn.Linear(512, 3)

    def forward(self, x):
        out = self.backbone(x)
        logits1 = self.fc1(out)
        logits2 = self.fc2(out)
        return logits1, logits2
model = WeatherModel()
model(paddle.to_tensor(np.random.rand(10, 3, 256, 256).astype(np.float32)))