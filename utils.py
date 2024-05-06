import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import torch

def preprocess(df, subseq):
  prot_list = []
  sequences = df.iloc[:, 1].values

  for i in tqdm(range(len(sequences))):
    len_seq = int(np.ceil(len(sequences[i]) / subseq))
    for idx in range(len_seq):
      if idx != len_seq - 1:
        prot_list.append(sequences[i][idx * subseq : (idx + 1) * subseq])
      else:
        prot_list.append(sequences[i][idx * subseq :])

  return prot_list

def get_embeddings(seq, tokenizer, model, device):
  batch_seq = [" ".join(list(seq))]
  ids = tokenizer(batch_seq)
  input_ids = torch.tensor(ids['input_ids']).to(device)
  attention_mask = torch.tensor(ids['attention_mask']).to(device)

  with torch.no_grad():
    embedding_repr = model(input_ids=input_ids, attention_mask=attention_mask)

  return embedding_repr.last_hidden_state[0].cpu().detach().numpy().mean(axis=0)

def preprocess_target(df, subseq):
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


def propagate_preds(predictions, ontologies_names, ontology):

  ont_n = ontologies_names.tolist()

  list_of_parents = []

  for idx_term in range(len(ont_n)):
    this_list_of_parents = []
    for parent in ontology[ont_n[idx_term]]['ancestors']:
      this_list_of_parents.append(ont_n.index(parent))
    list_of_parents.append(list(set(this_list_of_parents)))

  for idx_protein in tqdm(range(len(predictions))):
    for idx_term in range(len(ont_n)):
      for idx_parent in list_of_parents[idx_term]:
        predictions[idx_protein, idx_parent] = max(predictions[idx_protein, idx_parent], predictions[idx_protein, idx_term])
  return predictions

def evaluate(predictions, ground_truth, ontologies_names, ontology):

  predictions = propagate_preds(predictions, ontologies_names, ontology)

  f1_max_value = -1
  f1_max_threshold = -1

  for i in tqdm(range(1, 101)):
    threshold = i/100
    p, r = 0, 0
    number_of_proteins = 0

    for idx_protein in range(len(predictions)):
      protein_pred = set()
      protein_gt = set()

      protein_pred = ontologies_names[np.where(predictions[idx_protein, :] >= threshold)[0]]
      protein_gt = ontologies_names[np.where(ground_truth[idx_protein, :] == 1)[0]]

      if len(protein_pred) > 0:
        number_of_proteins += 1
        p += len(np.intersect1d(protein_pred, protein_gt)) / len(protein_pred) 
      r += len(np.intersect1d(protein_pred, protein_gt)) / len(protein_gt)

    if number_of_proteins > 0:
      threshold_p = p / number_of_proteins
    else:
      threshold_p = 0

    threshold_r = r / len(predictions)

    f1 = 0
    if threshold_p > 0 or threshold_r > 0:
      f1 = (2 * threshold_p * threshold_r) / (threshold_p + threshold_r)

    if f1 > f1_max_value:
      f1_max_value = f1
      f1_max_threshold = threshold

  print('F1 max:', f1_max_value)
  print('F1 threshold:', f1_max_threshold)


def get_ancestors(ontology, term):
  list_of_terms = []
  list_of_terms.append(term)
  data = []
  
  while len(list_of_terms) > 0:
    new_term = list_of_terms.pop(0)

    if new_term not in ontology:
      break
    data.append(new_term)
    for parent_term in ontology[new_term]['parents']:
      if parent_term in ontology:
        list_of_terms.append(parent_term)
  
  return data

def generate_ontology(file, specific_space=False, name_specific_space=''):
  ontology = {}
  gene = {}
  flag = False
  with open(file) as f:
    for line in f.readlines():
      line = line.replace('\n','')
      if line == '[Term]':
        if 'id' in gene:
          ontology[gene['id']] = gene
        gene = {}
        gene['parents'], gene['alt_ids'] = [], []
        flag = True
        
      elif line == '[Typedef]':
        flag = False
      
      else:
        if not flag:
          continue
        items = line.split(': ')
        if items[0] == 'id':
          gene['id'] = items[1]
        elif items[0] == 'alt_id':
          gene['alt_ids'].append(items[1])
        elif items[0] == 'namespace':
          if specific_space:
            if name_specific_space == items[1]:
              gene['namespace'] = items[1]
            else:
              gene = {}
              flag = False
          else:
            gene['namespace'] = items[1]
        elif items[0] == 'is_a':
          gene['parents'].append(items[1].split(' ! ')[0])
        elif items[0] == 'name':
          gene['name'] = items[1]
        elif items[0] == 'is_obsolete':
          gene = {}
          flag = False
    
    key_list = list(ontology.keys())
    for key in key_list:
      ontology[key]['ancestors'] = get_ancestors(ontology, key)
      for alt_ids in ontology[key]['alt_ids']:
        ontology[alt_ids] = ontology[key]
    
    for key, value in ontology.items():
      if 'children' not in value:
        value['children'] = []
      for p_id in value['parents']:
        if p_id in ontology:
          if 'children' not in ontology[p_id]:
            ontology[p_id]['children'] = []
          ontology[p_id]['children'].append(key)
    
  return ontology

def preprocess_blast(df, mode):
  seq = df.sequence.values
  id = 0
  fasta = ''
  for i in tqdm(seq):
    fasta += '>' + str(mode) + '_' + str(id) + '\n'
    id += 1
    fasta += i
    fasta += '\n'
  return fasta
