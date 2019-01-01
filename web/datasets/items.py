# -*- coding: utf-8 -*-

"""
 Functions to count the number of items
"""

# external imports
# ---
import numpy as np

# internal imports
# ---

from . import similarity
from . import analogy


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

    n = data.X.shape[0]

    # display a short sample of the data
    # ---

    limit = 5

    for i in range(limit):

        print(i + 1, X[i, 0], X[i, 1], y[i])

    print("---")

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


def count_msr_analogy():
    """
        Return the number of items in msr_analogy
        and display a sample of the data
        for checking purposes
    """

    data = analogy.fetch_msr_analogy()

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

    n = y.shape[0]

    print("---")

    print("number of items = ", n)

    return(n)
