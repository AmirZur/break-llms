# break-llms

## Reference

Code for

Petersen, Erika and Christopher Potts. 2022. [Lexical Semantics with Large Language Models: A Case Study of English _break_](https://ling.auf.net/lingbuzz/006859). Ms., Stanford University.


## Overview

1. `annotated_break_data.csv`: the annotated dataset

1. `annotated_dataset_study.ipynb` gets basic stats and tables for the annotated dataset

1. `static.ipynb`: static representations for _break_ in various versions of word2vec, GloVe, and fastText.

1. `get_all_reps.ipynb`: gets all the _break_ representations for all the models we consider. These representations are required for the notebooks `probing.ipynb` and `visualizations.ipynb`.

1. `probing.ipynb`: probing experiment code.

1. `visualizations.ipynb`: t-SNE-based visualizations of the _break_ representations.

1. `wordnet.ipynb`: basic analysis of the WordNet hypernym graph for _break_.

1. `break_utils.py`: helper code for many of the notebooks.

1. `fig`: directory containing visualizations included in the paper (output from `visualizations.ipynb` and `wordnet.ipynb`).

1. `reps`: directory in which representations are stored when `get_all_reps.ipynb` is run.

1. `results`: probing results files for the probes reported in the paper.