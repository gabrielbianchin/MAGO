import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import click
from utils import *

@click.command()
@click.option('--ont', '-ont', help="Ontology")

def main(ont):

  alpha = {'bp': 0.15, 'cc': 0.46, 'mf': 0.59}

  test = pd.read_csv('base/{}_test.csv'.format(ont))
  y_test = test.iloc[:, 2:].values
  ontologies_names = test.columns[2:].values

  preds_esm = np.load('predictions/mago-{}.npy'.format(ont))
  preds_blast = np.load('predictions/blast-{}.npy'.format(ont))

  if ont == 'bp':
    ontology = generate_ontology('base/go.obo', specific_space=True, name_specific_space='biological_process')
  elif ont == 'cc':
    ontology = generate_ontology('base/go.obo', specific_space=True, name_specific_space='cellular_component')
  else:
    ontology = generate_ontology('base/go.obo', specific_space=True, name_specific_space='molecular_function')

  preds = []
  for i, j in zip(preds_blast, preds_esm):
    if np.sum(i) != 0:
      preds.append(i * (1-alpha[ont]) + j * alpha[ont])
    else:
      preds.append(j)
  preds = np.array(preds)
  evaluate(preds, y_test, ontologies_names, ontology)
  np.save('predictions/mago+-{}.npy'.format(ont), preds)


if __name__ == '__main__':
  main()
