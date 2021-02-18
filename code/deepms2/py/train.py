# coding: utf-8
from __future__ import print_function
import os
import argparse

from pepms2 import PeptideMS2Options, PeptideMS2Trainer
from pepms2.utils import load_data_dir, save_data_json #, filter_data

parser = argparse.ArgumentParser(description='Predict fragment ion intensity')
parser.add_argument('--charge',type = str, default='charge2', help= 'charge type')
args = parser.parse_args()
options = PeptideMS2Options.default()


if args.charge == 'charge2':
  train_dir='./charge2'
else:
  train_dir='./charge3'
data, data_files = load_data_dir(train_dir)
# data, filter_indexs = filter_data(data, max_sequence_length=options.max_sequence_length, threshold=0.1)


os.makedirs(os.path.join(train_dir, 'model'), exist_ok=True)

trainer = PeptideMS2Trainer(
    options=options,
    save_path=os.path.join(train_dir, 'model', 'epoch_{epoch:03d}.hdf5'),
    log_path=os.path.join(train_dir, 'training.log')
)
result = trainer.train(data)
result['files'] = data_files
# result['filter'] = filter_indexs

trainer.save_model(os.path.join(train_dir, 'model', 'last_epoch.hdf5'))

save_data_json(result, os.path.join(train_dir, 'training.json'))

