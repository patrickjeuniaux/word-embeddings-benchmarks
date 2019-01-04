# -*- coding: utf-8 -*-

"""
 Functions to count the number of items
"""

# external imports
# ---
import numpy as np
from math import factorial
import importlib
import os
import six

# internal imports
# ---

import web.datasets.synonymy
import web.datasets.similarity
import web.datasets.analogy
import web.datasets.categorization

from utils import number_permutations


def coverage_BATS(vocabulary):
    """

    """

    import numpy as np

    data = web.datasets.analogy.fetch_BATS()

    X = data.X

    categories = data.category

    categories_high_level = data.category_high_level

    total_nb_items = 0

    total_nb_items_covered = 0

    for category in np.unique(categories):

        pairs = X[categories == category]

        nb_pairs = len(pairs)

        nb_items = number_permutations(2, nb_pairs)

        # convert numpy array to list of lists
        pairs = pairs.tolist()

        # we want to keep only the pairs covered

        # filter 1
        # ---
        candidates = [candidate for target_word, candidate in pairs
                      if target_word in vocabulary]

        # filter 2
        # ---
        final_candidates = []

        for candidate in candidates:

            nb_words_covered = 0

            if "/" not in candidate:

                if candidate in vocabulary:

                    nb_words_covered += 1

            else:

                words = candidate.split("/")

                for word in words:

                    if word in vocabulary:

                        nb_words_covered += 1

            if nb_words_covered > 0:

                final_candidates.append(candidate)

        nb_pairs_covered = len(final_candidates)

        nb_items_covered = number_permutations(2, nb_pairs_covered)

        print(category, ":", nb_pairs, "pairs (", nb_pairs_covered, "covered),", nb_items, "items (", nb_items_covered, "covered).")

        total_nb_items += nb_items

        total_nb_items_covered += nb_items_covered

    return coverage_result(total_nb_items, total_nb_items_covered)


def coverage_wordrep(vocabulary, subsample=None, rng=None):
    """

    """

    data = web.datasets.analogy.fetch_wordrep(subsample, rng)

    X = data.X

    categories = data.category

    categories_high_level = data.category_high_level

    wordnet_categories = data.wordnet_categories

    wikipedia_categories = data.wikipedia_categories

    # items counting
    # ---

    total_nb_items = 0

    total_nb_items_covered = 0

    for category in wordnet_categories | wikipedia_categories:

        pairs = X[categories == category]

        nb_pairs = len(pairs)

        nb_items = number_permutations(2, nb_pairs)

        # convert numpy array to list of lists
        pairs = pairs.tolist()

        # filter to keep only the pairs covered
        pairs = [(word1, word2) for word1, word2 in pairs
                 if word1 in vocabulary and word2 in vocabulary]

        nb_pairs_covered = len(pairs)

        nb_items_covered = number_permutations(2, nb_pairs_covered)

        print(category, ":", nb_pairs, "pairs (", nb_pairs_covered, "covered),", nb_items, "items (", nb_items_covered, "covered).")

        total_nb_items += nb_items

        total_nb_items_covered += nb_items_covered

    return coverage_result(total_nb_items, total_nb_items_covered)


def coverage_mikolov(corpus_name, vocabulary):
    """

    """

    # set the fetch function name
    # ---
    fetch_function_name = "fetch_" + corpus_name

    # retrieve the dataset
    # ---
    data = getattr(web.datasets.analogy, fetch_function_name)()

    question = data.X

    answer = data.y

    nb_items = answer.shape[0]

    nb_items_covered = 0

    for i in range(nb_items):

        word1, word2, word3 = question[i]

        if word1 in vocabulary and \
                word2 in vocabulary and \
                word3 in vocabulary and \
                answer[i] in vocabulary:

            nb_items_covered += 1

    return coverage_result(nb_items, nb_items_covered)


def coverage_similarity(dataset_name, vocabulary, **kwargs):
    """

    """

    # set the fetch function name
    # ---
    fetch_function_name = "fetch_" + dataset_name

    # retrieve the dataset
    # ---
    data = getattr(web.datasets.similarity, fetch_function_name)(**kwargs)

    # the word pair
    X = data.X

    nb_items = data.X.shape[0]

    nb_items_covered = 0

    for i in range(nb_items):

        word1, word2 = X[i]

        if word1 in vocabulary and word2 in vocabulary:

            nb_items_covered += 1

    return coverage_result(nb_items, nb_items_covered)


def coverage_synonymy(dataset_name, vocabulary):
    """

    """

    # set the fetch function name
    # ---
    fetch_function_name = "fetch_" + dataset_name

    # retrieve the dataset
    # ---
    data = getattr(web.datasets.synonymy, fetch_function_name)()

    # the question
    X = data.X

    # the answers
    y = data.y

    nb_items = data.X.shape[0]

    nb_items_covered = 0

    for i in range(nb_items):

        question = X[i]

        answers = y[i]

        good_answer = answers[0]
        bad_answers = answers[1:]

        # print(i + 1, X[i], y[i])

        bad_answers_covered = 0

        for bad_answer in bad_answers:

            if bad_answer in vocabulary:

                bad_answers_covered += 1

        if question in vocabulary and \
                good_answer in vocabulary and \
                bad_answers_covered > 0:

            nb_items_covered += 1

    return coverage_result(nb_items, nb_items_covered)


def coverage_categorization(dataset_name, vocabulary):
    """

    """

    # set the fetch function name
    # ---
    fetch_function_name = "fetch_" + dataset_name

    # retrieve the dataset
    # ---
    data = getattr(web.datasets.categorization, fetch_function_name)()

    # the question
    X = data.X

    nb_items = data.X.shape[0]

    nb_items_covered = 0

    for i in range(nb_items):

        word = X[i][0]

        if word in vocabulary:

            nb_items_covered += 1

    return coverage_result(nb_items, nb_items_covered)


def coverage_SAT(vocabulary):
    """

    """

    # retrieve the dataset
    # ---
    data = web.datasets.analogy.fetch_SAT()

    # the question
    X = data.X

    # the answers
    y = data.y

    nb_items = data.X.shape[0]

    nb_items_covered = 0

    def pair_in_vocab(pair, vocabulary):

        # if not a string (ex: if NaN)
        if not isinstance(pair, six.string_types):

            # https://pythonhosted.org/six/#constants

            return False

        word1, word2 = pair.split("_")

        return word1 in vocabulary and word2 in vocabulary

    for i in range(nb_items):

        question = X[i]

        answers = y[i]

        good_answer = answers[0]
        bad_answers = answers[1:]

        # print(i + 1, X[i], y[i])

        bad_answers_covered = 0

        for bad_answer in bad_answers:

            if pair_in_vocab(bad_answer, vocabulary):

                bad_answers_covered += 1

        if pair_in_vocab(question, vocabulary) and \
                pair_in_vocab(good_answer, vocabulary) and \
                bad_answers_covered > 0:

            nb_items_covered += 1

    return coverage_result(nb_items, nb_items_covered)


def coverage_result(nb_items, nb_items_covered):
    """

    """

    coverage = nb_items_covered / nb_items

    print("Number of items:", nb_items)
    print("Number of items covered:", nb_items_covered)
    print("Coverage:", round(coverage, 3))
    print("")

    results = (nb_items, nb_items_covered)

    return results


def load_vocabulary(input_path):
    """
    Return a Set of words found in the file
    - one word per line

    example:

    tax-exempt
    tax-exemption
    tax-free
    tax-funded
    tax-haven
    tax-payer
    tax-payers
    tax-paying
    tax-raising
    tax-rate
    tax-related
    tax-supported
    """

    print("Loading vocabulary")
    print("---")
    print("File:", input_path)

    vocabulary = set(line.strip() for line in open(input_path))

    print("---> loaded", len(vocabulary), "words")

    print("")

    return(vocabulary)


def calculate_coverage(vocabulary):
    """

    """

    print("Calculate coverage")
    print("---")

    # coverage_synonymy("TOEFL", vocabulary)
    # coverage_synonymy("ESL", vocabulary)

    # coverage_similarity("RG65", vocabulary)
    # coverage_similarity("RW", vocabulary)
    # coverage_similarity("SimLex999", vocabulary)
    # coverage_similarity("SimVerb3500", vocabulary)
    # coverage_similarity("WS353", vocabulary, which="all")
    # coverage_similarity("WS353", vocabulary, which="similarity")
    # coverage_similarity("WS353", vocabulary, which="relatedness")

    # coverage_similarity("MTurk", vocabulary)
    # coverage_similarity("MEN", vocabulary, which="all")

    # coverage_mikolov("msr_analogy", vocabulary)
    # coverage_mikolov("google_analogy", vocabulary)
    # coverage_wordrep(vocabulary)
    # coverage_BATS(vocabulary)
    # coverage_semeval_2012_2(vocabulary, "all")

    # coverage_SAT(vocabulary)

    # coverage_categorization("BLESS", vocabulary)

    # coverage_categorization("AP", vocabulary)

    # coverage_categorization("battig", vocabulary)
    # coverage_categorization("battig2010", vocabulary)

    # coverage_categorization("ESSLLI_1a", vocabulary)
    # coverage_categorization("ESSLLI_2b", vocabulary)
    # coverage_categorization("ESSLLI_2c", vocabulary)


if __name__ == "__main__":

    path = os.path.expanduser(os.path.join(
        "~", "Documents", "data", "DSM_eval", "5_vocabulary", "vocabulary.txt"))

    vocabulary = load_vocabulary(path)

    calculate_coverage(vocabulary)

    print("--- THE END ---")
