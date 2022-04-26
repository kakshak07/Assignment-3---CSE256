***[CSE 256 SP 2022: assignment 3: Comparing Language Models ]***





***[ Files ]***

There are three python files in this folder:

- (lm.py): This file describes the higher level interface for a language model, and contains functions to train, query, and evaluate it. An implementation of a simple back-off based unigram model is also included, that implements all of the functions
of the interface.

- (generator.py): This file contains a simple word and sentence sampler for any language model. Since it supports arbitarily complex language models, it is not very efficient. If this sampler is incredibly slow for your language model, you can consider implementing your own (by caching the conditional probability tables, for example, instead of computing it for every word).

-  (data.py): The primary file to run. This file contains methods to read the appropriate data files from the archive, train and evaluate all the unigram language models (by calling “lm.py”), and generate sample sentences from all the models (by calling  “generator.py”). It also saves the result tables into LaTeX files.

*** [ Tabulate ]***

The one *optional* dependency we have in this code is `tabulate` ([documentation](https://pypi.python.org/pypi/tabulate)), which you can install using a simple `pip install tabulate`.
This package is quite useful for generating the results table in LaTeX directly from your python code, which is a practice we encourage all of you to incorporate into your research as well.
If you do not install this package, the code does not write out the results to file (and you might get no runtime error).


*** [ Acknowledgements ]***
Python files adapted from a similar assignment by Sameer Singh