#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 This script evaluates all embeddings available in the package
 and saves .csv results

 Usage:

 ./evaluate_all_embeddings -o <output_dir>
"""

# external imports
# ---

from six import iteritems
from multiprocessing import Pool
from os import path
import logging
import optparse
import multiprocessing

# internal imports
# ---
from web.evaluate import evaluate_on_all
from web import embeddings


parser = optparse.OptionParser()

parser.add_option("-j", "--n_jobs", type="int", default=4)

parser.add_option("-o", "--output_dir", type="str", default="")

(opts, args) = parser.parse_args()

# Configure logging

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')

logger = logging.getLogger(__name__)


job_selection = []

job_selection.append('conceptnet')
# job_selection.append('FastText')
# job_selection.append('PDC')
# job_selection.append('HDC')
# job_selection.append('GloVe')
# job_selection.append('SG_GoogleNews')
# job_selection.append('LexVec')

# WARNING
# NMT embeddings are no longer available through the provided url
# job_selection.append('NMT')

jobs = []

# ConceptNet Numberbatch
# ---

if 'conceptnet' in job_selection:

    jobs.append(["fetch_conceptnet_numberbatch", {}])


# FastText
# ---

if 'FastText' in job_selection:

    jobs.append(["fetch_FastText", {}])

# PDC and HDC
# ---

for dim in [50, 100, 300]:

    if 'PDC' in job_selection:

        jobs.append(["fetch_PDC", {"dim": dim}])

    if 'HDC' in job_selection:

        jobs.append(["fetch_HDC", {"dim": dim}])

# GloVe
# ---

if 'GloVe' in job_selection:

    for dim in [50, 100, 200, 300]:

        jobs.append(["fetch_GloVe", {"dim": dim, "corpus": "wiki-6B"}])

    for dim in [25, 50, 100, 200]:

        jobs.append(["fetch_GloVe", {"dim": dim, "corpus": "twitter-27B"}])

    for corpus in ["common-crawl-42B", "common-crawl-840B"]:

        jobs.append(["fetch_GloVe", {"dim": 300, "corpus": corpus}])

# SG
# ---
if 'SG_GoogleNews' in job_selection:

    jobs.append(["fetch_SG_GoogleNews", {}])

# LexVec
# ---

if 'LexVec' in job_selection:

    jobs.append(["fetch_LexVec", {}])


# NMT
# ---

if 'NMT' in job_selection:

    # WARNING
    # NMT embeddings are no longer available through the provided url

    jobs.append(["fetch_NMT", {"which": "FR"}])
    jobs.append(["fetch_NMT", {"which": "DE"}])


def run_job(job):

    fetch_function_name, kwargs = job

    embedding_name = fetch_function_name.split("_")[1]

    filename = embedding_name

    if kwargs:

        job_description = "_" + "_".join(str(k) + "=" + str(v)
                                         for k, v in iteritems(kwargs))

        filename += job_description

    filename += ".csv"

    outf = path.join(opts.output_dir, filename)

    logger.info("Processing " + outf)

    if not path.exists(outf):

        w = getattr(embeddings, fetch_function_name)(**kwargs)

        res = evaluate_on_all_fast(w)

        res.to_csv(outf)


if __name__ == "__main__":

    Pool(opts.n_jobs).map(run_job, jobs)
