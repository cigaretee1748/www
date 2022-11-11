import pandas as pd

from dataReading import train_json

train_json['period'], period_dict = pd.factorize(train_json['period'])
train_json['weather'], weather_dict = pd.factorize(train_json['weather'])