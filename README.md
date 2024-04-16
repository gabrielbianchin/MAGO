# Integrating Transformers and AutoML for Protein Function Prediction

## Introduction
The next-generation sequencing technology and the decreasing cost of experimental verification of proteins have helped the accumulation of sequenced proteins in recent years. However, defining protein function is still difficult due to the cost and time required to this analysis. In consideration of that, computational methods using protein sequences have been developed to help decrease this gap. In this work, we present MAGO, an approach based on Transformers and AutoML, and MAGO+, an ensemble of MAGO with BLASTp, to deal with this task. MAGO and MAGO+ surpassed state-of-the-art methods based on machine learning and ensemble of local alignment tools with machine learning, improving the results based on Fmax and presenting statistically significant differences with the compared approaches.

The article is not yet available, but we will update the page as soon as it is accessible.

## Dataset
The dataset of this work can be found [here](https://zenodo.org/record/7409660).

## Reproducibility
* Create the folders ```embs```, ```models```, ```predictions```, and ```base```
* Unzip the dataset into the ```base``` folder
* For each ontology, run:
```
python extract.py -ont ontology
python train.py -ont ontology
python blast.py -ont ontology
python ensemble.py -ont ontology
```
The parameter **ontology** should be ```bp```, ```cc```, or ```mf``` for Biological Process (BP), Cellular Component (CC), or Molecular Function (MF), respectively.

## Citation
This repository contains the source codes of Integrating Transformers and AutoML for Protein Function Prediction, as given in the paper:

Gabriel Bianchin de Oliveira, Helio Pedrini, Zanoni Dias. "Integrating Transformers and AutoML for Protein Function Prediction", in proceedings of the 46th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC). Orlando, USA, 15 - 19 July 2024.

If you use this source code and/or its results, please cite our publication:
```
@inproceedings{OLIVEIRA_2024_EMBC,
  author = {G. B. Oliveira and H. Pedrini and Z. Dias},
  title = {Integrating Transformers and AutoML for Protein Function Prediction},
  booktitle = {46th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC)},
  address = {Orlando, USA},
  month = {jul},
  year = {2024}
}
```
