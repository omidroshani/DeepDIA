# coding: utf-8
from __future__ import print_function

import os
import re
import argparse

from pepms2 import PeptideMS2Options, PeptideMS2Predictor
from pepms2.modeling import build_model_from_weights
from pepms2.utils import save_data_json, load_peptides_csv

parser = argparse.ArgumentParser(description='Predict fragment ion intensity')
parser.add_argument('--charge',type = str, default='charge2', help= 'charge type')
args = parser.parse_args()

options = PeptideMS2Options.default()
predict_dir = '.'

if args.charge == 'charge2':
  charge='charge2'
else:
  charge='charge3'


for f in os.listdir(os.path.join(predict_dir,charge)):
  if re.match(r'^epoch_[0-9]+\.hdf5$', f) is not None:
      model_path =os.path.join(predict_dir, charge, f) 

predictor = PeptideMS2Predictor(options)
predictor.model = build_model_from_weights(options=options, weights_path=model_path)

data_files = [
    os.path.join(predict_dir,charge, f) 
    for f in os.listdir(os.path.join(predict_dir,charge)) 
    if f.endswith('.peptide.csv')
]

for file in data_files:
    
    peptides = load_peptides_csv(file)
    sequences = peptides['sequence']
    modifications = peptides['modification'] 
    
    prediction = predictor.predict(sequences, modifications)

    save_data_json(
        data=prediction,
        file=re.sub(r'\.peptide\.csv$', '.prediction.ions.json', file)
        #file=('prediction.ions.json')
    )
    

        
