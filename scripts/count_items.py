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


if __name__ == "__main__":

    # similarity
    # ---

    # count_similarity_items("RG65")
    # count_similarity_items("MTurk")
    # count_similarity_items("RW")
    # count_similarity_items("TR9856")
    # count_similarity_items("SimVerb3500")
    # count_similarity_items("SimLex999")

    # count_similarity_items("multilingual_SimLex999", which="EN")
    # count_similarity_items("multilingual_SimLex999", which="DE")
    # count_similarity_items("multilingual_SimLex999", which="IT")
    # count_similarity_items("multilingual_SimLex999", which="RU")

    # count_similarity_items("WS353", which="all")
    # count_similarity_items("WS353", which="relatedness")
    # count_similarity_items("WS353", which="similarity")

    # count_similarity_items("MEN", which="all")

    # analogy
    # ---
    # count_mikolov("msr_analogy")
    # count_mikolov("google_analogy")
    # count_semeval_2012_2("all")
    # count_wordrep()

    # categorization
    # ---

    pass
