#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Count the number of items in the datasets

 Usage:

 ./count_items

"""

# internal imports
# ---

from web.datasets.items import *


def count_similarity():
    """

    """

    assert count_similarity_items("RG65") == 65
    assert count_similarity_items("MTurk") == 287
    assert count_similarity_items("RW") == 2034
    assert count_similarity_items("TR9856") == 9856
    assert count_similarity_items("SimVerb3500") == 3500
    assert count_similarity_items("SimLex999") == 999

    assert count_similarity_items("multilingual_SimLex999", which="EN") == 999
    assert count_similarity_items("multilingual_SimLex999", which="DE") == 999
    assert count_similarity_items("multilingual_SimLex999", which="IT") == 999
    assert count_similarity_items("multilingual_SimLex999", which="RU") == 999

    assert count_similarity_items("WS353", which="all") == 353
    assert count_similarity_items("WS353", which="relatedness") == 252
    assert count_similarity_items("WS353", which="similarity") == 203

    assert count_similarity_items("MEN", which="all") == 3000


def count_analogy():
    """

    """

    assert count_mikolov("msr_analogy") == 8000
    assert count_mikolov("google_analogy") == 19544
    assert count_semeval_2012_2("all") == 3218
    assert count_wordrep() == 237409102
    assert count_SAT() == 374
    count_BATS() # 98000 != 99200 (PROBLEM)


def count_categorization():
    """

    """

    assert count_categorization_items("AP") == 402

    count_categorization_items("BLESS")  # 200 != 26554 (PROBLEM)
    count_categorization_items("battig")  # 5231 != 83 (PROBLEM)

    count_categorization_items("ESSLI_1a")  # 44
    count_categorization_items("ESSLI_2b")  # 40
    count_categorization_items("ESSLI_2c")  # 45
    # 44 + 40 + 45 = 129 != 134 (PROBLEM)


if __name__ == "__main__":

    # similarity
    # ---
    # count_similarity()

    # analogy
    # ---
    # count_analogy()

    # categorization
    # ---
    # count_categorization()
    pass
