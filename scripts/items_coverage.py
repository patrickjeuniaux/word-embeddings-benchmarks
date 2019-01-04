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

    path = os.path.expanduser(os.path.join(
        "~", "Documents", "data", "DSM_eval", "5_vocabulary", "vocabulary.txt"))

    vocabulary = load_vocabulary(path)

    calculate_coverage(vocabulary)

    print("--- THE END ---")
