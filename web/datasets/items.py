# -*- coding: utf-8 -*-

"""
 Functions to count the number of items
"""

# external imports
# ---
import numpy as np
from math import factorial

# internal imports
# ---

from . import similarity
from . import analogy
from . import categorization


def count_similarity_items(corpus_name, **kwargs):
    """
        Count the number of items in the similarity dataset
        and display a sample of the data for checking purposes
    """

    # dynamically set the fetch function name
    # ---
    fetch_function_name = "fetch_" + corpus_name

    # retrieve the dataset
    # ---
    data = getattr(similarity, fetch_function_name)(**kwargs)

    X = data.X

    y = data.y

    # display a short sample of the data
    # ---

    limit = 5

    for i in range(limit):

        print(i + 1, X[i, 0], X[i, 1], y[i])

    print("---")

    # items counting
    # ---

    n = data.X.shape[0]

    print("number of items = ", n)

    return(n)


def count_semeval_2012_2(which="all"):
    """
        Return the number of items in semeval_2012_2
        and display a sample of the data
        for checking purposes
    """

    data = analogy.fetch_semeval_2012_2(which)

    X_prot = data.X_prot

    X = data.X

    y = data.y

    categories_names = data.categories_names

    categories_descriptions = data.categories_descriptions

    # display a sample
    # ---
    categories = ('3_f', '8_f', '9_i')

    limit = 5

    for category in categories:

        print("")
        print(category)
        print("---")
        print("")
        print(categories_names[category])
        print(categories_descriptions[category])
        print(X_prot[category])
        print(X[category][:limit, :])
        print(y[category][:limit])

    # items counting
    # ---

    n = 0

    for category in categories_names:

        nb_questions = X[category].shape[0]

        n += nb_questions

    print("---")

    print("number of items = ", n)

    return(n)


def count_mikolov(corpus_name):
    """
        Return the number of items in msr_analogy or google_analogy
        and display a sample of the data
        for checking purposes
    """

    # dynamically set the fetch function name
    # ---
    fetch_function_name = "fetch_" + corpus_name

    # retrieve the dataset
    # ---
    data = getattr(analogy, fetch_function_name)()

    X = data.X

    y = data.y

    categories = data.category

    categories_high_level = data.category_high_level

    # display a sample
    # ---

    limit = 5

    print(X[:limit, :limit])
    print(y[:limit])
    print(categories[:limit])
    print(categories_high_level[:limit])

    print("---")

    # items counting
    # ---

    n = y.shape[0]

    print("number of items = ", n)

    return(n)


def number_permutations(k, n):
    """
    Calculate the number of permutations of k elements
    chosen in a set of n elements
    """
    return int(factorial(n) / factorial(n - k))


def count_wordrep(subsample=None, rng=None):
    """
        Return the number of items in wordrep
        and display a sample of the data
        for checking purposes
    """

    data = analogy.fetch_wordrep(subsample, rng)

    X = data.X

    categories = data.category

    categories_high_level = data.category_high_level

    wordnet_categories = data.wordnet_categories

    wikipedia_categories = data.wikipedia_categories

    # display a sample
    # ---

    limit = 5

    print(X[:limit])
    print(categories[:limit])
    print(categories_high_level[:limit])

    print("---")
    print("")
    print("WordNet categories:")
    print("---")
    print(wordnet_categories)
    print("")
    print("Wikipedia categories:")
    print("---")
    print(wikipedia_categories)

    # items counting
    # ---

    n = 0

    p1 = X.shape[0]

    p2 = 0

    print("")
    print("Statistics")
    print("---")
    print("")

    for category in wordnet_categories | wikipedia_categories:

        subX = X[categories == category]

        p = len(subX)

        np = number_permutations(2, p)

        print(category, " : ", p, " pairs, ", np, " permutations ")

        n += np

        p2 += p

    print("---")

    if p2 != p1:

        print ("Problem: p1 = ", p1, " != p2 = ", p2)

    print("number of words pairs = ", p1)

    print("number of items (i.e., number of permutations) = ", n)

    return(n)


def count_categorization_items(corpus_name, **kwargs):
    """
        Count the number of items in the categorization dataset
        and display a sample of the data for checking purposes
    """

    # dynamically set the fetch function name
    # ---
    fetch_function_name = "fetch_" + corpus_name

    # retrieve the dataset
    # ---
    data = getattr(categorization, fetch_function_name)(**kwargs)

    X = data.X

    y = data.y

    # display a short sample of the data
    # ---

    limit = 5

    for i in range(limit):

        print(i + 1, X[i], y[i])

    print("---")

    # items counting
    # ---

    n = data.X.shape[0]

    print("number of items = ", n)

    return(n)
