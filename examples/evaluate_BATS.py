# -*- coding: utf-8 -*-

"""
 Test BATS
"""

# external imports
# ---

import logging

# internal imports
# ---

from web.embeddings import fetch_GloVe
from web.datasets.analogy import fetch_BATS
from web.evaluate import evaluate_on_BATS

from web.analogy_solver import *

# Configure logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')


# Word embeddings
# ---

print("\nLoad embeddings")
print("Warning: it might take a few minutes")
print("---")

# Fetch GloVe embedding
# ---
words_embedding = fetch_GloVe(corpus="wiki-6B", dim=50)


# evaluation on BATS
# ---

print("\nLaunch evaluation on BATS")
print("Warning: it will take a few minutes")
print("---")


df = evaluate_on_BATS(words_embedding)

print(df)
