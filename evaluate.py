import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import torch

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