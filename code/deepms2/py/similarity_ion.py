# coding: utf-8
from __future__ import print_function

import os
import re

from pepms2 import PeptideMS2Options, PeptideMS2Predictor
from pepms2.modeling import build_model_from_weights
from pepms2.utils import save_data_json, load_peptides_csv
from pepms2.training import split_train_validate
from pepms2.utils import load_data_dir, save_data_json
from pepms2.preprocessing import DataConverter
from pepms2.modeling import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Predict fragment ion intensity')
parser.add_argument('--charge',type = str, default='charge2', help= 'charge type')
args = parser.parse_args()

if args.charge == 'charge2':
  charge='charge2'
else:
  charge='charge3'

train_dir = '.'
train_dir = os.path.join(train_dir ,charge)

data, data_files = load_data_dir(train_dir)
options=PeptideMS2Options.default()
converter = DataConverter(options)


x, y = converter.data_to_tensor(data)
seed=0
validate_percent=.33
x_train, y_train, x_validate, y_validate, train_indexs, validate_indexs = split_train_validate(x, y, validate_percent=validate_percent, seed=seed)




predict_dir = '.'
for f in os.listdir(os.path.join(predict_dir,charge)):
  if re.match(r'^epoch_[0-9]+\.hdf5$', f) is not None:
      model_path =os.path.join(predict_dir, charge, f) 


predictor = PeptideMS2Predictor(options)
predictor.model = build_model_from_weights(options=options, weights_path=model_path)
y_pred = predictor.predict_test(x_validate)

y_pred = y_pred.astype('float32')
y_validate = y_validate.astype('float32')

p = cosine_similarity(y_validate, y_pred)
print("\nmedian of arr, axis = None : ", np.median(p))
#violin plot of cosine_similarity matrix
p1 = p
p1 = p1*(-1)
p1 = np.reshape(p1, p1.shape[0]*p1.shape[1])
plt.violinplot(p1, showmedians=True)
plt.ylabel("dot product")

plt.savefig('cosine_similarity_hele2.png')
plt.show()


    

      
