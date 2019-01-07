# -*- coding: utf-8 -*-

"""
 Simple example illustrating the evaluation of an embedding on similarity datasets
"""

# external imports
# ---

import logging
from six import iteritems

# internal imports
# ---

from web.datasets.similarity import fetch_MEN, fetch_WS353, fetch_SimLex999
from web.embeddings import fetch_GloVe
from web.evaluate import evaluate_similarity
# import sys


print("\nEvaluate similarity")
print("---")

# Configure logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')

print("\n1. Load embeddings")
print("Warning: it might take few minutes")
print("---")

# Fetch GloVe embedding (warning: it might take few minutes)
w_glove = fetch_GloVe(corpus="wiki-6B", dim=300)

print("\n2. Define tasks")
print("---")
# Define tasks
tasks = {
    "MEN": fetch_MEN(),
    "WS353": fetch_WS353(),
    "SimLex999": fetch_SimLex999(),
}

# sys.exit()

print("\n3. Print sample data")
print("---")
# Print sample data
for name, data in iteritems(tasks):

    print("Sample data from {}: pair \"{}\" and \"{}\" is assigned score {}".
          format(name, data.X[0][0], data.X[0][1], data.y[0]))

print("\n4. Calculate results")
print("---")
# Calculate results using helper function
for name, data in iteritems(tasks):

    print("Spearman correlation of scores on {} {}".
          format(name, evaluate_similarity(w_glove, data.X, data.y)))

print("\n--- THE END ---")
