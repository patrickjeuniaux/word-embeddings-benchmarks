#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Count the number of items in the datasets

 Usage:

 python items_coverage.py

"""

# external imports
# ---

import os

# internal imports
# ---

from web.datasets.items.coverage import *


if __name__ == "__main__":

    # IO
    # ---

    vocabulary_path = os.path.expanduser(os.path.join(
        "~", "Documents", "data", "DSM_eval", "5_vocabulary", "vocabulary.txt"))

    results_folder = os.path.expanduser(os.path.join(
        "~", "Documents", "data", "DSM_eval", "4_coverage"))

    attend_output_folder(results_folder)

    results_path = os.path.join(results_folder, "items_coverage.txt")

    # load vocabulary
    # ---
    vocabulary = load_vocabulary(vocabulary_path)

    # calculate coverage
    # ---

    calculate_coverage(vocabulary, results_path)
