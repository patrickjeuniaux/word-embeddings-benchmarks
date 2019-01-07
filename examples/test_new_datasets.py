# -*- coding: utf-8 -*-

"""
 Test of a variety of new datasets
"""

# external imports
# ---

import logging
from six import iteritems

# internal imports
# ---

from web.embeddings import fetch_GloVe

from web.evaluate import evaluate_similarity
from web.evaluate import evaluate_categorization

from web.datasets.similarity import fetch_SimVerb3500
from web.datasets.analogy import fetch_BATS
from web.datasets.categorization import fetch_battig2010


# Configure logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')

# datasets of interest
# ---

datasets = []
# datasets.append("SimVerb3500")
# datasets.append("battig2010")
# datasets.append("battig2010")


# Word embeddings
# ---

if datasets != []:

    print("\nLoad embeddings")
    print("Warning: it might take few minutes")
    print("---")

    # Fetch GloVe embedding (warning: it might take few minutes)
    w_glove = fetch_GloVe(corpus="wiki-6B", dim=50)
    # w_glove = fetch_GloVe(corpus="wiki-6B", dim=300)

else:
    print("\nThere are no datasets to evaluate...")


# SimVerb3500
# ---

if "SimVerb3500" in datasets:

    print("\nSimVerb3500")
    print("---")

    data = fetch_SimVerb3500()

    print("\nSample of the data:")
    print("---")
    for i in range(5):

        print(i + 1, data.X[i], data.y[i])

    print("\nEvaluation of similarity:")
    print("---")

    correlation = evaluate_similarity(w_glove, data.X, data.y)

    print("\nSpearman correlation = ", correlation)


# Battig2010
# ---

if "battig2010" in datasets:

    print("\nbattig2010")
    print("---")

    data = fetch_battig2010()

    print("\nSample of the data:")
    print("---")
    for i in range(5):

        print(i + 1, data.X[i], data.y[i])

    print("\nEvaluation of categorization:")
    print("---")

    purity = evaluate_categorization(w_glove, data.X, data.y)

    print("\nCluster purity = ", purity)


# BATS
# ---


print("\nBATS")
print("---")

data = fetch_BATS()

categories = ('UK_city - county', 'adj - comparative', 'antonyms - gradable')


print("\nSample of the data:")
print("---")

for category in categories:

    print("")
    print(category)
    X = data.X[data.category == category]

    for i in range(5):

        print(i + 1, X[i])

print("\nEvaluation of categorization:")
print("---")

# purity = evaluate_categorization(w_glove, data.X, data.y)

# print("\nCluster purity = ", purity)


print("\n--- THE END ---")
