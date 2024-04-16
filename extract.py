import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from transformers import AutoModel, AutoTokenizer
import torch
import sys
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm
import click
from utils import *

@click.command()
@click.option('--ont', '-ont', help="Ontology")

def main(ont):
  
  model_name = 'facebook/esm2_t36_3B_UR50D'
  window = 1024
  embed_size = 2560

  train = preprocess(df=pd.read_csv('base/' + ont + '_train.csv'), subseq=window-2)
  val = preprocess(df=pd.read_csv('base/' + ont + '_val.csv'), subseq=window-2)
  test = preprocess(df=pd.read_csv('base/' + ont + '_val.csv'), subseq=window-2)

  tokenizer = AutoTokenizer.from_pretrained(model_name)
  device = torch.device("cuda:0")
  model = AutoModel.from_pretrained(model_name).to(device)
  model.eval()

  # training
  embeds = np.zeros((len(train), embed_size))
  i = 0
  for this_seq in tqdm(train):
    embeds[i] = get_embeddings(this_seq)
    i += 1
    gc.collect()
    torch.cuda.empty_cache()
  np.save('embs/' + ont + '-train.npy', embeds)

  # validation
  embeds = np.zeros((len(val), embed_size))
  i = 0
  for this_seq in tqdm(val):
    embeds[i] = get_embeddings(this_seq)
    i += 1
    gc.collect()
    torch.cuda.empty_cache()
  np.save('embs/' + ont + '-val.npy', embeds)

  # test
  embeds = np.zeros((len(test), embed_size))
  i = 0
  for this_seq in tqdm(test):
    embeds[i] = get_embeddings(this_seq)
    i += 1
    gc.collect()
    torch.cuda.empty_cache()
  np.save('embs/' + ont + '-test.npy', embeds)


if __name__ == '__main__':
  main()