# -*- coding: utf-8 -*-

"""
 Test of a variety of new datasets
"""

# external imports
# ---

import logging
from six import iteritems
import os

# internal imports
# ---

from web.embeddings import fetch_GloVe
from web.embeddings import load_embedding

from web.evaluate import evaluate_similarity
from web.evaluate import evaluate_categorization
from web.evaluate import evaluate_on_WordRep
from web.evaluate import evaluate_on_BATS
from web.evaluate import evaluate_on_SAT
from web.evaluate import evaluate_on_synonyms
from web.evaluate import evaluate_on_semeval_2012_2


from web.datasets.similarity import fetch_SimVerb3500
from web.datasets.categorization import fetch_battig2010


# Configure logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')

# datasets of interest
# ---

datasets = []
# datasets.append("WordRep")
# datasets.append("SimVerb3500")
# datasets.append("battig2010")
# datasets.append("BATS")
# datasets.append("SAT")
# datasets.append("TOEFL")
# datasets.append("ESL")
datasets.append("SemEval")


# Word embeddings
# ---

if datasets != []:

    print("\nLoad embeddings")
    print("Warning: it might take a few minutes")
    print("---")

    # Fetch GloVe embedding
    # ---

    # w = fetch_GloVe(corpus="wiki-6B", dim=50)
    # w = fetch_GloVe(corpus="wiki-6B", dim=300)

    input_path = '~/web_data/embeddings/glove.6B/glove.6B.50d.SAMPLE.txt'
    fname = os.path.expanduser(input_path)

    load_kwargs = {"vocab_size": 50000, "dim": 50}

    w = load_embedding(fname=fname,
                       format="glove",
                       normalize=True,
                       lower=False,
                       clean_words=False,
                       load_kwargs=load_kwargs)

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

    results = evaluate_similarity(w, data.X, data.y)

    correlation = results['correlation']
    nb_items = results['count']
    missing_words = results['missing']

    print("\nSpearman correlation = ", correlation)
    print("Number of items = ", nb_items)
    print("Missing words = ", missing_words)


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

    results = evaluate_categorization(w, data.X, data.y)

    purity = results['purity']

    nb_items = results['count']

    missing = results['missing']

    print("\nCluster purity = ", purity)
    print("nb items = ", nb_items)
    print("Missing = ", missing)


if "WordRep" in datasets:

    # evaluation on WordRep
    # ---
    # limit to 50 pairs (-> 50*49 = 2450 permutations to test)
    # ---

    max_pairs = 50

    print("\nLaunch evaluation on WordRep")
    print("Warning: it will take a long time even for small values of 'max_pairs'")
    print("---")
    print("max_pairs=", max_pairs)

    df = evaluate_on_WordRep(w, max_pairs=max_pairs)

    print(df)


# BATS
# ---

if "BATS" in datasets:

    print("\nBATS")
    print("Warning: it will take a few minutes")
    print("---")

    df = evaluate_on_BATS(w)

    print(df)


# SAT
# ---

if "SAT" in datasets:

    print("\nLaunch evaluation on SAT")
    print("---")

    df = evaluate_on_SAT(w)

    print(df)


# TOEFL
# ---

if "TOEFL" in datasets:

    print("\nLaunch evaluation on TOEFL")
    print("---")

    df = evaluate_on_synonyms(w, "TOEFL")

    print(df)


# ESL
# ---

if "ESL" in datasets:

    print("\nLaunch evaluation on ESL")
    print("---")

    df = evaluate_on_synonyms(w, "ESL")

    print(df)

# SemEval2012
# ---

if "SemEval" in datasets:

    print("\nLaunch evaluation on SemEval 2012")
    print("---")

    results = evaluate_on_semeval_2012_2(w)

    print(results)


print("\n--- THE END ---")
