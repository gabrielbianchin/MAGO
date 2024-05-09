import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import autokeras as ak
import click
from evaluate import *

gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

@click.command()
@click.option('--ont', '-ont', help="Ontology")

def preprocess(df, subseq):
  y = []
  positions = []
  sequences = df.iloc[:, 1].values
  target = df.iloc[:, 2:].values
  for i in tqdm(range(len(sequences))):
    len_seq = int(np.ceil(len(sequences[i]) / subseq))
    for idx in range(len_seq):
      y.append(target[i])
      positions.append(i)
  return np.array(y), np.array(positions)

def emb_method(matrix_embs, method):
  if method == 'mean':
    return np.mean(matrix_embs, axis=0)

def protein_embedding(X, y, pos, method='mean'):
  n_X = []
  last_pos = pos[0]
  cur_emb = []
  n_y = [y[0]]
  for i in range(len(X)):
    cur_pos = pos[i]
    if last_pos == cur_pos:
      cur_emb.append(X[i])
    else:
      n_X.append(emb_method(np.array(cur_emb), method))
      last_pos = cur_pos
      cur_emb = [X[i]]
      n_y.append(y[i])
  n_X.append(emb_method(np.array(cur_emb), method))

  return np.array(n_X), np.array(n_y)

def main(ont):

  window = 1024

  y_train, pos_train = preprocess(df=pd.read_csv('base/{}_train.csv'.format(ont)), subseq=window-2)
  y_val, pos_val = preprocess(df=pd.read_csv('base/{}_val.csv'.format(ont)), subseq=window-2)
  y_test, pos_test = preprocess(df=pd.read_csv('base/{}_test.csv'.format(ont)), subseq=window-2)
  ontologies_names = pd.read_csv('base/{}_val.csv'.format(ont)).columns[2:].values

  X_train = np.load('embs/{}-train.npy'.format(ont))
  X_val = np.load('embs/{}-val.npy'.format(ont))
  X_test = np.load('embs/{}-test.npy'.format(ont))

  X_train, y_train = protein_embedding(X_train, y_train, pos_train)
  X_val, y_val = protein_embedding(X_val, y_val, pos_val)
  X_test, y_test = protein_embedding(X_test, y_test, pos_test)


  clf = ak.StructuredDataClassifier(multi_label=True, metrics='binary_accuracy', objective='val_loss', max_trials=50, project_name='automl-{}'.format(ont))

  es = tf.keras.callbacks.EarlyStopping(patience=5, verbose=1, restore_best_weights=True)
  clf.fit(X_train, y_train, validation_data=(X_val, y_val), callbacks=[es], verbose=1, epochs=20)

  model = clf.export_model()

  preds = model.predict(X_test)

  if ont == 'bp':
    ontology = generate_ontology('base/go.obo', specific_space=True, name_specific_space='biological_process')
  elif ont == 'cc':
    ontology = generate_ontology('base/go.obo', specific_space=True, name_specific_space='cellular_component')
  else:
    ontology = generate_ontology('base/go.obo', specific_space=True, name_specific_space='molecular_function')

  evaluate(preds, y_test, ontologies_names, ontology)
  model.save('models/best-model-{}.h5'.format(ont))
  np.save('predictions/mago-{}.npy'.format(ont), preds)

if __name__ == '__main__':
  main()
