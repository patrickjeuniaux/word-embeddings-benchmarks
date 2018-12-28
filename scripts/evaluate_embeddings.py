#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 This script evaluates all embeddings available in the package
 and saves .csv results

 Usage:

 ./evaluate_embeddings -o <output_dir>
"""
from web.evaluate import evaluate_on_all
from web import embeddings
from six import iteritems
from multiprocessing import Pool
from os import path
import logging
import optparse
import multiprocessing

parser = optparse.OptionParser()
parser.add_option("-j", "--n_jobs", type="int", default=4)
parser.add_option("-o", "--output_dir", type="str", default="")
(opts, args) = parser.parse_args()

# Configure logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')
logger = logging.getLogger(__name__)

jobs = []


# ConceptNet Numberbatch
# ---

jobs.append(["fetch_conceptnet_numberbatch", {}])

# FastText
# ---

jobs.append(["fetch_FastText", {}])

# PDC and HDC
# ---

for dim in [50, 100, 300]:

    jobs.append(["fetch_PDC", {"dim": dim}])

    jobs.append(["fetch_HDC", {"dim": dim}])

# GloVe
# ---

for dim in [50, 100, 200, 300]:

    jobs.append(["fetch_GloVe", {"dim": dim, "corpus": "wiki-6B"}])

for dim in [25, 50, 100, 200]:

    jobs.append(["fetch_GloVe", {"dim": dim, "corpus": "twitter-27B"}])


for corpus in ["common-crawl-42B", "common-crawl-840B"]:

    jobs.append(["fetch_GloVe", {"dim": 300, "corpus": corpus}])

# SG
# ---

jobs.append(["fetch_SG_GoogleNews", {}])

# LexVec
# ---

jobs.append(["fetch_LexVec", {}])


# NMT
# ---

# NMT embeddings are not longer available thru the provided url

# jobs.append(["fetch_NMT", {"which": "FR"}])
# jobs.append(["fetch_NMT", {"which": "DE"}])



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

        res = evaluate_on_all(w)

        res.to_csv(outf)


if __name__ == "__main__":

    Pool(opts.n_jobs).map(run_job, jobs)
