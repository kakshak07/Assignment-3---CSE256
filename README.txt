***[CSE 256 SP 2022: assignment 3: Comparing Language Models ]***




Files:

There are 4 python files

There are three python files in this folder:

- (lm.py): This file describes the higher level interface for a language model, and contains functions to train, query, and evaluate it. An implementation of a simple back-off based unigram model is also included, that implements all of the functions of the interface. In this file I have implemented one class for Trigram + Laplace Smoothing.

- (generator.py): This file contains a simple word and sentence sampler for any language model. Since it supports arbitarily complex language models, it is not very efficient. If this sampler is incredibly slow for your language model, you can consider implementing your own (by caching the conditional probability tables, for example, instead of computing it for every word).

-  (data.py): The primary file to run. In this file it contains different functions to run the code. If you want to vary training data, please use this function def learn_variable_unigram() and pass the training data. If you want to run unigram model use def learn_ungram() function for unigram model. To implement trigram + Laplace smoothing model there is a function def learn_trigram(), to build trigram model. This file builds model, internally calls generator sampler to generate sentences, it also contains analysis of in-domain as well as out-of domain model analysis.

- adaptationBonus.py: This file contains code for training model on full dataset A and partial fraction of dataset B, you can vary the size of partial dataset to compare models.

There is jupyter notebook file for Visualization and Analysis of graphs. It contains all line plot, heat map and plots to visualize and get a comprehensive view of the project.
