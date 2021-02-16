# coding: utf-8

from __future__ import print_function

import pandas as pd
from scipy.stats import pearsonr
from peprt import PeptideRTPredictor
from peprt.models import max_sequence_length
from peprt.trainer import data_to_tensors
import matplotlib.pyplot as plt
import numpy as np
import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap

import itertools
import json
import os
import re


def load_data(file):
    with open(file, 'r') as f:
        data = pd.read_csv(file)
    return data

def load_data_dir(dir):
    filenames = [        
        f for f in os.listdir(dir) 
        if f.endswith(".irt.csv")
    ]
    data = [load_data(os.path.join(dir, f)) for f in filenames]
    data = pd.concat(data)
    return data, filenames

def using_mpl_scatter_density(fig, x, y):
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    density = ax.scatter_density(x, y, cmap=white_viridis)
    ax.title.set_text('Correlation = ' + "{:.2f}".format(corr*100))
    fig.colorbar(density, label='iRT')

def split_train_validate(x, y, validate_percent=.33, seed=None):
    length = len(x)
    np.random.seed(seed)
    indexs = np.random.permutation(length)    
    train_end = int((1 - validate_percent) * length)
    train_indexs = indexs[:train_end]
    validate_indexs = indexs[train_end:]
    x_train = x[train_indexs]
    y_train = y[train_indexs]
    x_validate =x[validate_indexs]
    y_validate =y[validate_indexs]
    return x_train, y_train, x_validate, y_validate, train_indexs, validate_indexs

#read all data and spliti it to train and test set
train_dir = './irt'
data, data_files = load_data_dir(train_dir)
rt_min=-50
rt_max=150
x, y = data_to_tensors(data, rt_min=rt_min, rt_max=rt_max)
x_train, y_train, x_validate, y_validate, train_indexs, validate_indexs = split_train_validate(x, y, validate_percent=0.33, seed=0)

#define model path for our trained model
#model_path = "./irt_model/last_epoch.hdf5"
predict_dir = '.'
model_path = [ 
    os.path.join(predict_dir, 'irt', f) 
    for f in os.listdir(os.path.join(predict_dir, 'irt'))
    if re.match(r'^epoch_[0-9]+\.hdf5$', f) is not None
][-1]

#define model path for repo trained model
#model_path = "./irt_model/epoch_082.hdf5"
predictor = PeptideRTPredictor()
predictor.load_weights(model_path)
 
y_pred = predictor.predict_test(x_validate)
mask_pred = y_validate>-50
y_validate = y_validate[mask_pred]
y_pred = y_pred[mask_pred]


#computing pearson_coff
y_pred_list = np. reshape(y_pred,y_pred.shape[0])
y_validate_list = np. reshape(y_validate,y_validate.shape[0])
corr, _ = pearsonr(y_pred_list, y_validate_list)
print("pearson_coff",corr)


# "Viridis-like" colormap with white background
white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-20, '#440053'),
    (0.2, '#404388'),
    (0.4, '#2a788e'),
    (0.6, '#21a784'),
    (0.8, '#78d151'),
    (1, '#fde624'),
], N=256)


fig = plt.figure()
using_mpl_scatter_density(fig, y_pred, y_validate)
plt.xlabel("Predicted iRT")
plt.ylabel("Experimental iRT")

plt.savefig('irt_hele2.png')
plt.show()
     
