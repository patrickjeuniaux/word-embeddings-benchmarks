# -*- coding: utf-8 -*-

"""
 Test WordRep
"""

# external imports
# ---

import logging

# internal imports
# ---

from web.embeddings import fetch_GloVe
from web.datasets.analogy import fetch_wordrep
from web.evaluate import evaluate_on_WordRep

from web.analogy_solver import *

# Configure logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')


# Word embeddings
# ---

print("\nLoad embeddings")
print("Warning: it might take few minutes")
print("---")

# Fetch GloVe embedding (warning: it might take few minutes)
# ---
words_embedding = fetch_GloVe(corpus="wiki-6B", dim=50)


# evaluation on WordRep
# ---
# limit to 50 pairs (-> 50*49 = 2450 permutations to test)
# ---

df = evaluate_on_WordRep(words_embedding, max_pairs=50)

print(df)
