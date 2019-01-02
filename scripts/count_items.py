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


def count_synonymy():
    """

    """
    assert count_Xy_items("synonymy", "ESL") == 50
    assert count_Xy_items("synonymy", "TOEFL") == 80


def count_similarity():
    """

    """

    assert count_Xy_items("similarity", "RG65") == 65
    assert count_Xy_items("similarity", "MTurk") == 287
    assert count_Xy_items("similarity", "RW") == 2034
    assert count_Xy_items("similarity", "TR9856") == 9856
    assert count_Xy_items("similarity", "SimVerb3500") == 3500
    assert count_Xy_items("similarity", "SimLex999") == 999

    assert count_Xy_items("similarity", "multilingual_SimLex999", which="EN") == 999
    assert count_Xy_items("similarity", "multilingual_SimLex999", which="DE") == 999
    assert count_Xy_items("similarity", "multilingual_SimLex999", which="IT") == 999
    assert count_Xy_items("similarity", "multilingual_SimLex999", which="RU") == 999

    assert count_Xy_items("similarity", "WS353", which="all") == 353
    assert count_Xy_items("similarity", "WS353", which="relatedness") == 252
    assert count_Xy_items("similarity", "WS353", which="similarity") == 203

    assert count_Xy_items("similarity", "MEN", which="all") == 3000


def count_analogy():
    """

    """

    assert count_mikolov("msr_analogy") == 8000
    assert count_mikolov("google_analogy") == 19544
    assert count_semeval_2012_2("all") == 3218
    assert count_wordrep() == 237409102
    assert count_Xy_items("analogy", "SAT") == 374
    assert count_BATS() == 98000


def count_categorization():
    """

    """

    assert count_Xy_items("categorization", "AP") == 402

    assert count_Xy_items("categorization", "BLESS") == 200
    assert count_Xy_items("categorization", "battig") == 5231

    assert count_Xy_items("categorization", "ESSLI_1a") == 44
    assert count_Xy_items("categorization", "ESSLI_2b") == 40
    assert count_Xy_items("categorization", "ESSLI_2c") == 45


if __name__ == "__main__":

    # count_synonymy()
    # count_similarity()
    # count_analogy()
    # count_categorization()

    # analogy
    # ---
    count_BATS()  # 98000 != 99200 (PROBLEM)

    # categorization
    # ---
    count_Xy_items("categorization", "BLESS")  # 200 != 26554 (PROBLEM)
    count_Xy_items("categorization", "battig")  # 5231 != 83 (PROBLEM)

    n1a = count_Xy_items("categorization", "ESSLI_1a")  # 44
    n2b = count_Xy_items("categorization", "ESSLI_2b")  # 40
    n2c = count_Xy_items("categorization", "ESSLI_2c")  # 45
    # 44 + 40 + 45 = 129 != 134 (PROBLEM)

    pass
