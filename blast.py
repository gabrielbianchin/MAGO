import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import sys
import click
from utils import *

@click.command()
@click.option('--ont', '-ont', help="Ontology")

def main(ont):
  
  train = pd.read_csv('base/' + ont + '_train.csv')
  test = pd.read_csv('base/' + ont + '_test.csv')

  seq_train = preprocess_blast(base_treino, 'train')
  seq_test = preprocess_blast(base_teste, 'test')

  with open('base/reference-' + ont + '.fasta', 'w') as f:
    print(seq_train, file=f)

  with open('base/queries-' + ont + '.fasta', 'w') as f:
    print(seq_test, file=f)

  os.system('makeblastdb -in "base/reference-' + ont + '.fasta" -dbtype prot')

  seq = {}
  s=''
  k=''
  with open('queries-' + ont + '.fasta', "r") as f:
    for lines in f:
      if lines[0]==">":
        if s!='':
          seq[k] = s
          s=''
        k = lines[1:].strip('\n')
      else:
        s+=lines.strip('\n')
  seq[k] = s

  output = os.popen('blastp -db base/reference-' + ont + '.fasta -query base/queries-' + ont + '.fasta -outfmt "6 qseqid sseqid bitscore" -evalue 0.001').readlines()

  with open('base/output-' + ont + '.txt', 'w') as f:
    print(''.join(output), file=f)

  with open('base/output-' + ont + '.txt') as f:
    output = f.readlines()

  test_bits={}
  test_train={}
  for lines in output:
    line = lines.strip('\n').split()
    try:
      if float(line[2]) >= 250:
        if line[0] in test_bits:
          test_bits[line[0]].append(float(line[2]))
          test_train[line[0]].append(line[1])
        else:
          test_bits[line[0]] = [float(line[2])]
          test_train[line[0]] = [line[1]]
    except:
      pass

  preds_score=[]
  nlabels = len(y_test[0])

  for s in seq:
    probs = np.zeros(nlabels, dtype=np.float32)
    if s in test_bits:
      weights = np.array(test_bits[s])/np.sum(test_bits[s])

      for j in range(len(test_train[s])):
        id = int(test_train[s][j].split('_')[1])
        temp = y_train[id]
        probs+= weights[j] * temp

    preds_score.append(probs)

  preds_score = np.array(preds_score)
  np.save('predictions/blast-' + ont + '.npy', preds)


if __name__ == '__main__':
  main()