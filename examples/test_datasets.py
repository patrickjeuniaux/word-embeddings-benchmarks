# -*- coding: utf-8 -*-

"""
 Test of a variety of new datasets
"""

# external imports
# ---

import logging
from six import iteritems
import os

# internal imports
# ---

from web.embeddings import *

from web.evaluate import *

from web.datasets.synonymy import *
from web.datasets.similarity import *
from web.datasets.analogy import *
from web.datasets.categorization import *

# Configure logging
# ---

# logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')


datasets = []

# Synonyms tasks
# ---

datasets.append("TOEFL")  # *new*
datasets.append("ESL")  # *new*

# Similarity tasks
# ---

datasets.append("MEN")
datasets.append("WS353")
datasets.append("WS353S")
datasets.append("WS353R")
datasets.append("SimLex999")
datasets.append("RW")
datasets.append("RG65")
datasets.append("MTurk")
datasets.append("TR9856")
datasets.append("SimVerb3500")  # *new*

# Analogy tasks
# ---

datasets.append("Google")
datasets.append("MSR")
datasets.append("SemEval")
datasets.append("WordRep")
datasets.append("SAT")

# Categorization tasks
# ---

datasets.append("AP")
datasets.append("BLESS")
datasets.append("Battig")
datasets.append("battig2010")  # *new*
datasets.append("ESSLLI_1a")
datasets.append("ESSLLI_2b")
datasets.append("ESSLLI_2c")
datasets.append("BATS")  # *new*


def test_embedding_on_datasets(w):
    '''

    '''

    if 'ESSLLI_1a' in datasets:

        data = fetch_ESSLLI_1a()

        results = evaluate_categorization(w, data.X, data.y)

        print(results)

    if 'ESSLLI_2b' in datasets:

        data = fetch_ESSLLI_2b()

        results = evaluate_categorization(w, data.X, data.y)

        print(results)

    if 'ESSLLI_2c' in datasets:

        data = fetch_ESSLLI_2c()

        results = evaluate_categorization(w, data.X, data.y)

        print(results)

    if 'Battig' in datasets:

        data = fetch_battig()

        results = evaluate_categorization(w, data.X, data.y)

        print(results)

    if 'BLESS' in datasets:

        data = fetch_BLESS()

        results = evaluate_categorization(w, data.X, data.y)

        print(results)

    if 'AP' in datasets:

        data = fetch_AP()

        results = evaluate_categorization(w, data.X, data.y)

        print(results)

    if "Google" in datasets:

        data = fetch_google_analogy()

        # categories = np.unique(data.category)

        # print(categories)

        results = evaluate_analogy(w, data.X, data.y, category=data.category)

        print(results)

    if "MSR" in datasets:

        data = fetch_msr_analogy()

        # categories = np.unique(data.category)

        # print(categories)

        results = evaluate_analogy(w, data.X, data.y, category=data.category)

        print(results)

    if "WS353S" in datasets:

        data = fetch_WS353(which="similarity")

        results = evaluate_similarity(w, data.X, data.y)

        print(results)

    if "WS353R" in datasets:

        data = fetch_WS353(which="relatedness")

        results = evaluate_similarity(w, data.X, data.y)

        print(results)

    if "WS353" in datasets:

        data = fetch_WS353()

        results = evaluate_similarity(w, data.X, data.y)

        print(results)

    if "MEN" in datasets:

        data = fetch_MEN()

        results = evaluate_similarity(w, data.X, data.y)

        print(results)

    if "TR9856" in datasets:

        data = fetch_TR9856()

        results = evaluate_similarity(w, data.X, data.y)

        print(results)

    if "MTurk" in datasets:

        data = fetch_MTurk()

        results = evaluate_similarity(w, data.X, data.y)

        print(results)

    if "SimLex999" in datasets:

        data = fetch_SimLex999()

        results = evaluate_similarity(w, data.X, data.y)

        print(results)

    if "RW" in datasets:

        data = fetch_RW()

        results = evaluate_similarity(w, data.X, data.y)

        print(results)

    if "RG65" in datasets:

        data = fetch_RG65()

        results = evaluate_similarity(w, data.X, data.y)

        print(results)

    if "SemEval" in datasets:

        results = evaluate_on_semeval_2012_2(w)

        print(results)

    # SimVerb3500 : similarity / Spearman correlation
    # ---

    if 'SimVerb3500' in datasets:

        data = fetch_SimVerb3500()

        # print("\nSample of the data:")

        # print("---")
        # for i in range(5):

        #     print(i + 1, data.X[i], data.y[i])

        results = evaluate_similarity(w, data.X, data.y)

        print(results)
    #    correlation  count  missing
    # 0     0.140293   3500      232

    # Battig2010 : categorization / purity
    # ---

    if 'battig2010' in datasets:

        data = fetch_battig2010()

        results = evaluate_categorization(w, data.X, data.y)

        print(results)

    #      purity  count  missing
    # 0  0.768293     82        6

    # ESL : synonyms / accuracy
    # ---

    if 'ESL' in datasets:

        results = evaluate_on_synonyms(w, "ESL")

        print(results)

    #    accuracy  correct  count  missing
    # 0       0.4       20     50        0

    # TOEFL : synonyms / accuracy
    # ---

    if 'TOEFL' in datasets:

        results = evaluate_on_synonyms(w, "TOEFL")

        print(results)

    #    accuracy  correct  count  missing
    # 0     0.525       42     80        3

    # SAT : analogy / accuracy
    # ---

    if 'SAT' in datasets:

        results = evaluate_on_SAT(w)

        print(results)

    #    accuracy  correct  count  missing
    # 0  0.213904       80    374      816

    # BATS : categorization / accuracy
    # ---

    if 'BATS' in datasets:

        # Warning: it will take a few minutes"
        # ---

        results = evaluate_on_BATS(w)

        print(results)

    #                       accuracy  correct  count  missing
    # verb_ving - 3psg      0.271429    665.0   2450        0
    # hypernyms - animals   0.021951     36.0   1640        0
    # verb+ment_irreg       0.089855    186.0   2070        0
    # synonyms - intensity  0.033245     75.0   2256        0
    # verb_3psg - ved       0.349796    857.0   2450        0
    # meronyms - substance  0.020352     44.0   2162        0
    # noun+less_reg         0.000000      0.0    240        0
    # verb_inf - ved        0.439592   1077.0   2450        0
    # adj+ly_reg            0.101633    249.0   2450        0
    # animal - sound        0.000000      0.0    812        0
    # re+verb_reg           0.048913     27.0    552        0
    # verb+tion_irreg       0.105263    148.0   1406        0
    # country - capital     0.714286   1750.0   2450        0
    # antonyms - binary     0.118409    256.0   2162        0
    # name - nationality    0.066463    109.0   1640        0
    # synonyms - exact      0.049202    111.0   2256        0
    # uk_city - county      0.027926     63.0   2256        0
    # hypernyms - misc      0.022648     39.0   1722        0
    # antonyms - gradable   0.053191    120.0   2256        0
    # meronyms - member     0.018039     39.0   2162        0
    # animal - young        0.012179     19.0   1560        0
    # adj - comparative     0.642857    117.0    182        0
    # over+adj_reg          0.018382      5.0    272        0
    # adj+ness_reg          0.131868     24.0    182        0
    # verb_inf - ving       0.488980   1198.0   2450        0
    # animal - shelter      0.001742      3.0   1722        0
    # verb_inf - 3psg       0.493469   1209.0   2450        0
    # un+adj_reg            0.093061    228.0   2450        0
    # verb_ving - ved       0.426122   1044.0   2450        0
    # verb+er_irreg         0.038415     63.0   1640        0
    # noun - plural_reg     0.570612   1398.0   2450        0
    # hyponyms - misc       0.024155     50.0   2070        0
    # country - language    0.112245    264.0   2352        0
    # noun - plural_irreg   0.335459    789.0   2352        0
    # male - female         0.385714    486.0   1260        0
    # verb+able_reg         0.033069     25.0    756        0
    # adj - superlative     0.383041    131.0    342        0
    # meronyms - part       0.016908     35.0   2070        0
    # things - color        0.048566    105.0   2162        0
    # name - occupation     0.142512    295.0   2070        0

    if 'WordRep' in datasets:

        # Warning: it will take a long time even for small values of 'max_pairs'
        results = evaluate_on_WordRep(w, max_pairs=10)

        print(results)

        #                        accuracy  correct  count  missing
        # nationality-adjective  0.955556     86.0     90        0
        # present-participle     0.288889     26.0     90       27
        # Attribute              0.000000      0.0     90        9
        # Causes                 0.000000      0.0     90       18
        # man-woman              0.111111     10.0     90        0
        # IsA                    0.000000      0.0     90       54
        # PartOf                 0.044444      4.0     90       45
        # adjective-to-adverb    0.077778      7.0     90       18
        # Antonym                0.000000      0.0     90      108
        # DerivedFrom            0.000000      0.0     90       54
        # currency               0.055556      5.0     90        0
        # SimilarTo              0.000000      0.0     90       54
        # MadeOf                 0.000000      0.0     90       54
        # comparative            0.288889     26.0     90       27
        # plural-nouns           0.466667     42.0     90        0
        # HasContext             0.000000      0.0     90       36
        # InstanceOf             0.000000      0.0     90        0
        # plural-verbs           0.211111     19.0     90        9
        # past-tense             0.300000     27.0     90       27
        # superlative            0.111111     10.0     90       18
        # RelatedTo              0.000000      0.0     90       45
        # Entails                0.000000      0.0     90        0
        # city-in-state          0.000000      0.0     90       72
        # all-capital-cities     0.555556     50.0     90        0
        # MemberOf               0.000000      0.0     90      108
        # wikipedia              0.003419      4.0   1170      585
        # all                    0.138667    312.0   2250      783
        # wordnet                0.285185    308.0   1080      198

    print("--- THE END ---")


if __name__ == "__main__":

    w = load_toy_embedding()

    test_embedding_on_datasets(w)
