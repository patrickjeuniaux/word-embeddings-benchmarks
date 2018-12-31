# -*- coding: utf-8 -*-

"""
 Functions to count the number of items
"""

# external imports
# ---
import numpy as np

# internal imports
# ---

from .similarity import *
from .analogy import *



def count_similarity_items(data):
    """
        Count the number of items in the similarity dataset
        and display a sample of the data for checking purposes
    """

    X = data.X

    y = data.y

    n = data.X.shape[0]

    limit = 5

    for i in range(limit):

        print(i + 1, X[i, 0], X[i, 1], y[i])

    print("---")

    print("number of items = ", n)

    return(n)




def count_RG65():
    """
        Return the number of items in RG65
    """

    data = fetch_RG65()

    n = count_similarity_items(data)

    return(n)


def count_MTurk():
    """
        Return the number of items in MTurk
    """

    data = fetch_MTurk()

    n = count_similarity_items(data)

    return(n)


def count_RW():
    """
        Return the number of items in RW
    """

    data = fetch_RW()

    n = count_similarity_items(data)

    return(n)


def count_TR9856():
    """
        Return the number of items in TR9856
    """

    data = fetch_TR9856()

    n = count_similarity_items(data)

    return(n)


def count_SimVerb3500():
    """
        Return the number of items in SimVerb3500
    """

    data = fetch_SimVerb3500()

    n = count_similarity_items(data)

    return(n)


def count_SimLex999():
    """
        Return the number of items in SimLex999
    """

    data = fetch_SimLex999()

    n = count_similarity_items(data)

    return(n)


def count_multilingual_SimLex999(which="EN"):
    """
        Return the number of items in multilingual_SimLex999
    """

    data = fetch_multilingual_SimLex999(which)

    n = count_similarity_items(data)

    return(n)


def count_WS353(which="all"):
    """
        Return the number of items in WS353
    """

    data = fetch_WS353(which)

    n = count_similarity_items(data)

    return(n)


def count_MEN(which="all"):
    """
        Return the number of items in MEN
    """

    data = fetch_MEN(which)

    n = count_similarity_items(data)

    return(n)


def count_semeval_2012_2(which="all"):
    """
        Return the number of items in semeval_2012_2
        and display a sample of the data
        for checking purposes
    """

    data = fetch_semeval_2012_2(which)

    X_prot = data.X_prot

    X = data.X

    y = data.y

    categories_names = data.categories_names

    categories_descriptions = data.categories_descriptions

    # excerpt
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
        print(X[category][:limit,:])
        print(y[category][:limit])

    # items counting
    # ---

    n = 0

    for category in categories_names:

        nb_prototypes = X_prot[category].shape[0]
        nb_questions = X[category].shape[0]

        n += nb_prototypes

        n += nb_questions

    print("---")

    print("number of items = ", n)

    return(n)










