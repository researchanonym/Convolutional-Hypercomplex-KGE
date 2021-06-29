# Convolutional Hypercomplex Embeddings for Link Prediction
This open-source project contains the Pytorch implementation of our approaches (QMult, OMult, ConvQ and ConvO), training and evaluation scripts.
To foster further reproducible research and alleviate hardware requirements to reproduce the reported results, we provide pretrained models on all datasets.

## Installation

First clone the repository:
```
https://github.com/researchanonym/Convolutional-Hypercomplex-KGE
```
Then obtain the required libraries:
```
conda env create -f environment.yml
conda activate hypercomplex
```
The code is compatible with Python 3.6.4

## Reproducing reported results
- ```unzip KGs.zip```.
- Download pretrained models (1.8 GB) via [Google Drive](https://drive.google.com/file/d/1qhOoccJlAMMe4FLO4LamjM9KwlCJ9UQx/view?usp=sharing).
- ```unzip PretrainedModels.zip```  
- Reproduce reported link prediction results: ``` python reproduce_link_prediction_results.py```
- Reproduce reported link prediction results based on only tail entity rankings: ``` python reproduce_link_prediction_results_based_on_tail_entity_rankings.py```
- Reproduce reported link prediction per relation results: ``` python reproduce_link_prediction_per_relation.py```
