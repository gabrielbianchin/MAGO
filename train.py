import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import autokeras as ak
import click
from utils import *

gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

@click.command()
@click.option('--ont', '-ont', help="Ontology")

def main(ont):

  window = 1022

  y_train, pos_train = preprocess_target(df=pd.read_csv('base/' + ont + '_train.csv'), subseq=window-2)
  y_val, pos_val = preprocess_target(df=pd.read_csv('base/' + ont + '_val.csv'), subseq=window-2)
  y_test, pos_test = preprocess_target(df=pd.read_csv('base/' + ont + '_val.csv'), subseq=window-2)
  ontologies_names = pd.read_csv('base/' + ont + '_val.csv').columns[2:].values

  X_train = np.load('embs/' + ont + '-train.npy')
  X_val = np.load('embs/' + ont + '-val.npy')
  X_test = np.load('embs/' + ont + '-test.npy')

  X_train, y_train = protein_embedding(X_train, y_train, pos_train)
  X_val, y_val = protein_embedding(X_val, y_val, pos_val)
  X_test, y_test = protein_embedding(X_test, y_test, pos_test)


  with tf.device('/gpu:0'):
    clf = ak.StructuredDataClassifier(multi_label=True, metrics='binary_accuracy', objective='val_loss')

    es = tf.keras.callbacks.EarlyStopping(patience=5, verbose=1, restore_best_weights=True)
    clf.fit(X_train, y_train, validation_data=(X_val, y_val), callbacks=[es], verbose=1, epochs=30)

    model = clf.export_model()

    preds = model.predict(X_test)

    if ont == 'bp':
      ontology = generate_ontology('base/go.obo', specific_space=True, name_specific_space='biological_process')
    elif ont == 'cc':
      ontology = generate_ontology('base/go.obo', specific_space=True, name_specific_space='cellular_component')
    else:
      ontology = generate_ontology('base/go.obo', specific_space=True, name_specific_space='molecular_function')

    evaluate(preds, y_test, ontologies_names, ontology)
    model.save('models/best-model-' + ont +'.h5')
    np.save('predictions/mago-' + ont + '.npy', preds)


if __name__ == '__main__':
  main()