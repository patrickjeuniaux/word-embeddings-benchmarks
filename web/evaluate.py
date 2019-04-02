# -*- coding: utf-8 -*-
"""
 Evaluation functions
"""

# external imports
# ---

import logging
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from six import iteritems
import six
import numpy as np
import importlib
import os

# internal imports
# ---

import web.datasets.synonymy

from web.datasets.similarity import fetch_MEN
from web.datasets.similarity import fetch_WS353
from web.datasets.similarity import fetch_SimLex999
from web.datasets.similarity import fetch_MTurk
from web.datasets.similarity import fetch_RG65
from web.datasets.similarity import fetch_RW
from web.datasets.similarity import fetch_TR9856
from web.datasets.similarity import fetch_SimVerb3500

from web.datasets.categorization import fetch_AP
from web.datasets.categorization import fetch_battig
from web.datasets.categorization import fetch_battig2010
from web.datasets.categorization import fetch_BLESS
from web.datasets.categorization import fetch_ESSLLI_1a
from web.datasets.categorization import fetch_ESSLLI_2b
from web.datasets.categorization import fetch_ESSLLI_2c

# import of analogy datasets fetchers
# and many other things (ex: itertools.product)
# are accomplished within analogy_solver
# ---
from web.analogy_solver import *

from web.embedding import Embedding
from web.embeddings import load_toy_embedding


def evaluate_on_all(w):
    """
    Evaluate Embedding w on all benchmarks

    Parameters
    ----------
    w: Embedding or dict
      Embedding to evaluate.

    Returns
    -------
    results: pandas.DataFrame
      DataFrame with results, one per column.
    """
    if isinstance(w, dict):

        w = Embedding.from_dict(w)

    # Synonyms tasks
    # ---

    synonymy_datasets = []

    synonymy_datasets.append("TOEFL")  # *new*
    synonymy_datasets.append("ESL")  # *new*

    # Similarity tasks
    # ---

    similarity_datasets = []

    # similarity_datasets.append("MEN")
    # similarity_datasets.append("WS353")
    # similarity_datasets.append("WS353S")
    # similarity_datasets.append("WS353R")
    # similarity_datasets.append("SimLex999")
    # similarity_datasets.append("RW")
    # similarity_datasets.append("RG65")
    # similarity_datasets.append("MTurk")
    # similarity_datasets.append("TR9856")
    similarity_datasets.append("SimVerb3500")  # *new*

    # Analogy tasks
    # ---

    analogy_datasets = []

    analogy_datasets.append("Google")
    # analogy_datasets.append("MSR")
    # analogy_datasets.append("SemEval")
    # analogy_datasets.append("WordRep")
    # analogy_datasets.append("SAT")

    # Categorization tasks
    # ---

    categorization_datasets = []

    # categorization_datasets.append("AP")
    # categorization_datasets.append("BLESS")
    # categorization_datasets.append("battig")
    # categorization_datasets.append("battig2010")  # *new*
    # categorization_datasets.append("ESSLLI_1a")
    # categorization_datasets.append("ESSLLI_2b")
    # categorization_datasets.append("ESSLLI_2c")
    # categorization_datasets.append("BATS")  # *new*

    # Calculate results on synonymy
    # ---

    logger.info("\nCalculating synonymy benchmarks")

    results = {}

    for dataset in synonymy_datasets:

        df = evaluate_on_synonyms(w, dataset)

        msg = "\nResults for {}\n---\n{}".format(dataset, df)

        logger.info(msg)

        df['task'] = 'synonymy'
        df['dataset'] = dataset

        results[dataset] = df

    # Calculate results on similarity
    # ---

    logger.info("\nCalculating similarity benchmarks")

    for dataset in similarity_datasets:

        if dataset == 'WS353R':

            mydataset = 'WS353'

            kwargs = {'which': 'relatedness'}

        elif dataset == 'WS353S':

            mydataset = 'WS353'

            kwargs = {'which': 'similarity'}

        else:
            mydataset = dataset

            kwargs = {}

        fetch_function_name = "fetch_" + mydataset
        module = importlib.import_module("web.datasets.similarity")
        data = getattr(module, fetch_function_name)(**kwargs)

        df = evaluate_similarity(w, data.X, data.y)

        msg = "\nResults for {}\n---\n{}".format(dataset, df)

        logger.info(msg)

        df['dataset'] = dataset
        df['task'] = 'similarity'

        results[dataset] = df

    # Calculate results on analogy
    # ---

    logger.info("\nCalculating analogy benchmarks")

    for dataset in analogy_datasets:

        if dataset == "Google":

            data = fetch_google_analogy()
            df = evaluate_analogy(w, data.X, data.y, category=data.category)

            df['dataset'] = dataset

        elif dataset == "MSR":

            data = fetch_msr_analogy()
            df = evaluate_analogy(w, data.X, data.y, category=data.category)

            df['dataset'] = dataset

        elif dataset == "SemEval":

            df = evaluate_on_semeval_2012_2(w)

        elif dataset == "SAT":

            df = evaluate_on_SAT(w)

        elif dataset == "WordRep":

            df = evaluate_on_WordRep(w, max_pairs=10)

        else:

            continue

        msg = "\nResults for {}\n---\n{}".format(dataset, df)

        logger.info(msg)

        results[dataset] = df

    # Calculate results on categorization
    # ---

    logger.info("\nCalculating categorization benchmarks")

    for dataset in categorization_datasets:

        if dataset == 'BATS':

            result = evaluate_on_BATS(w)

        else:

            fetch_function_name = "fetch_" + dataset

            module = importlib.import_module("web.datasets.categorization")

            data = getattr(module, fetch_function_name)()

            result = evaluate_categorization(w, data.X, data.y)

        msg = "\nResults for {}\n---\n{}".format(dataset, result)

        logger.info(msg)

        results[dataset] = result

    # Construct pandas table
    # ---

    dfs = None

    for dataset, df in results.items():

        print(dataset)
        print("---")

        df.reset_index(inplace=True)

        print(df)
        print(df.shape)

        if dfs is None:

            dfs = df

        else:

            dfs = pd.concat([dfs, df], axis=0, ignore_index=True)

    return dfs


def evaluate_on_all_fast(w):
    """
    Evaluate Embedding w on all fast-running benchmarks

    Parameters
    ----------
    w: Embedding or dict
      Embedding to evaluate.

    Returns
    -------
    results: pandas.DataFrame
      DataFrame with results, one per column.
    """
    if isinstance(w, dict):

        w = Embedding.from_dict(w)

    # Calculate results on similarity
    # ---

    logger.info("Calculating similarity benchmarks")

    similarity_tasks = {
        "MEN": fetch_MEN(),
        "WS353": fetch_WS353(),
        "WS353R": fetch_WS353(which="relatedness"),
        "WS353S": fetch_WS353(which="similarity"),
        "SimLex999": fetch_SimLex999(),
        "RW": fetch_RW(),
        "RG65": fetch_RG65(),
        "MTurk": fetch_MTurk(),
        "TR9856": fetch_TR9856(),
        "SimVerb3500": fetch_SimVerb3500(),
    }

    similarity_results = {}

    for name, data in iteritems(similarity_tasks):

        # compute Spearkan correlation
        # ---
        similarity_results[name] = evaluate_similarity(w, data.X, data.y)

        logger.info("Spearman correlation of scores on {} {}".format(name, similarity_results[name]))

    # Calculate results on analogy
    # ---

    logger.info("Calculating analogy benchmarks")

    analogy_tasks = {
        "Google": fetch_google_analogy(),
        "MSR": fetch_msr_analogy()
    }

    analogy_results = {}

    for name, data in iteritems(analogy_tasks):

        analogy_results[name] = evaluate_analogy(w, data.X, data.y)

        logger.info("Analogy prediction accuracy on {} {}".format(name, analogy_results[name]))

    SemEval = evaluate_on_semeval_2012_2(w)

    for k in SemEval:

        analogy_results[k] = SemEval[k]

    logger.info("Analogy prediction accuracy on {} {}".format("SemEval2012", analogy_results["SemEval2012_2"]))

    # Calculate results on categorization

    logger.info("Calculating categorization benchmarks")

    categorization_tasks = {
        "AP": fetch_AP(),
        "BLESS": fetch_BLESS(),
        "Battig": fetch_battig(),
        "ESSLLI_2c": fetch_ESSLLI_2c(),
        "ESSLLI_2b": fetch_ESSLLI_2b(),
        "ESSLLI_1a": fetch_ESSLLI_1a()
    }

    categorization_results = {}

    # Calculate results using helper function

    for name, data in iteritems(categorization_tasks):

        categorization_results[name] = evaluate_categorization(w, data.X, data.y)

        logger.info("Cluster purity on {} {}".format(name, categorization_results[name]))

    # Construct pd table

    cat = pd.DataFrame([categorization_results])

    analogy = pd.DataFrame([analogy_results])

    sim = pd.DataFrame([similarity_results])

    results = cat.join(sim).join(analogy)

    return results


def evaluate_similarity(w, X, y):
    """
    Calculate Spearman correlation between cosine similarity of the model
    and human rated similarity of word pairs

    Parameters
    ----------
    w : Embedding or dict
      Embedding or dict instance.

    X: array, shape: (n_samples, 2)
      Word pairs

    y: vector, shape: (n_samples,)
      Human ratings

    Returns
    -------
    cor: float
      Spearman correlation
    """

    if isinstance(w, dict):

        w = Embedding.from_dict(w)

    missing_words = 0

    words = w.vocabulary.word_id

    for query in X:

        for query_word in query:

            if query_word not in words:

                missing_words += 1

    if missing_words > 0:

        logger.warning("Missing {} words. Will replace them with mean vector".format(missing_words))

    mean_vector = np.mean(w.vectors, axis=0, keepdims=True)

    A = np.vstack(w.get(word, mean_vector) for word in X[:, 0])

    B = np.vstack(w.get(word, mean_vector) for word in X[:, 1])

    scores = np.array([v1.dot(v2.T) / (np.linalg.norm(v1) * np.linalg.norm(v2)) for v1, v2 in zip(A, B)])

    correlation = scipy.stats.spearmanr(scores, y).correlation

    nb_items = len(y)

    data = [pd.Series(correlation, name="correlation"),
            pd.Series(nb_items, name="count"),
            pd.Series(missing_words, name="missing")]

    results = pd.concat(data, axis=1)

    return results


def calculate_purity(y_true, y_pred):
    """
    Calculate purity for given true and predicted cluster labels.

    Parameters
    ----------
    y_true: array, shape: (n_samples, 1)
      True cluster labels

    y_pred: array, shape: (n_samples, 1)
      Cluster assingment.

    Returns
    -------
    purity: float
      Calculated purity.
    """
    assert len(y_true) == len(y_pred)

    true_clusters = np.zeros(shape=(len(set(y_true)), len(y_true)))

    pred_clusters = np.zeros_like(true_clusters)

    for id, cl in enumerate(set(y_true)):

        true_clusters[id] = (y_true == cl).astype("int")

    for id, cl in enumerate(set(y_pred)):

        pred_clusters[id] = (y_pred == cl).astype("int")

    M = pred_clusters.dot(true_clusters.T)

    purity = 1. / len(y_true) * np.sum(np.max(M, axis=1))

    nb_items = len(y_true)

    results = {'purity': purity,
               'nb_items': nb_items}

    return results


def evaluate_categorization(w, X, y, method="all", seed=None):
    """
    Evaluate embeddings on categorization task.

    Parameters
    ----------
    w: Embedding or dict
      Embedding to test.

    X: vector, shape: (n_samples, )
      Vector of words.

    y: vector, shape: (n_samples, )
      Vector of cluster assignments.

    method: string, default: "all"
      What method to use. Possible values are "agglomerative", "kmeans", "all.
      If "agglomerative" is passed, method will fit AgglomerativeClustering
      (with very crude hyperparameter tuning to avoid overfitting).
      If "kmeans" is passed, method will fit KMeans.
      In both cases number of clusters is preset to the correct value.

    seed: int, default: None
      Seed passed to KMeans.

    Returns
    -------
    purity: float
      Purity of the best obtained clustering.

    Notes
    -----
    KMedoids method was excluded as empirically didn't improve over KMeans (for categorization
    tasks available in the package).
    """

    if isinstance(w, dict):

        w = Embedding.from_dict(w)

    assert method in ["all", "kmeans", "agglomerative"], "Uncrecognized method"

    mean_vector = np.mean(w.vectors, axis=0, keepdims=True)

    words = np.vstack(w.get(word, mean_vector) for word in X.flatten())

    missing_words = sum([1 for word in X.flatten() if word not in w])
    # print("missing words:", missing_words)

    ids = np.random.RandomState(seed).choice(range(len(X)), len(X), replace=False)

    # Evaluate clustering on several hyperparameters of AgglomerativeClustering and
    # KMeans
    best_purity = 0

    if method == "all" or method == "agglomerative":

        results = calculate_purity(y[ids], AgglomerativeClustering(n_clusters=len(set(y)),
                                                                   affinity="euclidean",
                                                                   linkage="ward").fit_predict(words[ids]))

        best_purity = results['purity']

        logger.debug("Purity={:.3f} using affinity={} linkage={}".format(best_purity, 'euclidean', 'ward'))

        for affinity in ["cosine", "euclidean"]:

            for linkage in ["average", "complete"]:

                results = calculate_purity(y[ids], AgglomerativeClustering(n_clusters=len(set(y)),
                                                                           affinity=affinity,
                                                                           linkage=linkage).fit_predict(words[ids]))
                purity = results['purity']

                logger.debug("Purity={:.3f} using affinity={} linkage={}".format(purity, affinity, linkage))

                best_purity = max(best_purity, purity)

    if method == "all" or method == "kmeans":

        results = calculate_purity(y[ids], KMeans(random_state=seed, n_init=10, n_clusters=len(set(y))).
                                   fit_predict(words[ids]))

        purity = results['purity']

        logger.debug("Purity={:.3f} using KMeans".format(purity))

        best_purity = max(purity, best_purity)

    nb_items = len(y)

    data = [pd.Series(best_purity, name="purity"),
            pd.Series(nb_items, name="count"),
            pd.Series(missing_words, name="missing")]

    results = pd.concat(data, axis=1)

    return results


def evaluate_analogy(w, X, y, method="add", k=None, category=None, batch_size=100):
    """
    Simple method to score embedding using SimpleAnalogySolver

    used with MSR and GOOGLE datasets

    Other datasets use other evaluation methods

    Parameters
    ----------
    w : Embedding or dict
      Embedding or dict instance.

    method : {"add", "mul"}
      Method to use when finding analogy answer, see "Improving Distributional Similarity
      with Lessons Learned from Word Embeddings"

    X : array-like, shape (n_samples, 3)
      Analogy questions.

    y : array-like, shape (n_samples, )
      Analogy answers.

    k : int, default: None
      If not None will select k top most frequent words from embedding

    batch_size : int, default: 100
      Increase to increase memory consumption and decrease running time

    category : list, default: None
      Category of each example, if passed function returns accuracy per category
      in addition to the overall performance.
      Analogy datasets have "category" field that can be supplied here.

    Returns
    -------
    result: dict
      Results, where each key is for given category and special empty key "" stores
      summarized accuracy across categories
    """
    if isinstance(w, dict):

        w = Embedding.from_dict(w)

    assert category is None or len(category) == y.shape[0], "Passed incorrect category list"

    solver = SimpleAnalogySolver(w=w, method=method, batch_size=batch_size, k=k)

    predictions = solver.predict(X)

    # print(predictions)

    y_pred = predictions['predictions']

    accuracy = OrderedDict({"all": np.mean(y_pred == y)})

    count = OrderedDict({"all": len(y_pred)})

    correct = OrderedDict({"all": np.sum(y_pred == y)})

    if category is not None:

        for cat in set(category):

            accuracy[cat] = np.mean(y_pred[category == cat] == y[category == cat])

            count[cat] = np.sum(category == cat)

            correct[cat] = np.sum(y_pred[category == cat] == y[category == cat])

    df = pd.concat([pd.Series(accuracy, name="accuracy"),
                    pd.Series(correct, name="correct"),
                    pd.Series(count, name="count")],
                   axis=1)

    df['category'] = df.index

    df['task'] = 'analogy'

    return df


def evaluate_on_semeval_2012_2(w):
    """
    Simple method to score embedding
    Note: it is NOT using SimpleAnalogySolver
    but another method

    Parameters
    ----------
    w : Embedding or dict
      Embedding or dict instance.

    Returns
    -------
    result: pandas.DataFrame
      Results with spearman correlation per broad category with special key "all" for summary
      spearman correlation
    """
    if isinstance(w, dict):

        w = Embedding.from_dict(w)

    data = fetch_semeval_2012_2()

    mean_vector = np.mean(w.vectors, axis=0, keepdims=True)

    categories = data.y.keys()

    results = defaultdict(list)

    for c in categories:

        # Get mean of left and right vector
        # ---

        prototypes = data.X_prot[c]

        prot_left = np.mean(np.vstack(w.get(word, mean_vector) for word in prototypes[:, 0]), axis=0)

        prot_right = np.mean(np.vstack(w.get(word, mean_vector) for word in prototypes[:, 1]), axis=0)

        questions = data.X[c]

        question_left, question_right = np.vstack(w.get(word, mean_vector) for word in questions[:, 0]), \
            np.vstack(w.get(word, mean_vector) for word in questions[:, 1])

        scores = np.dot(prot_left - prot_right, (question_left - question_right).T)

        c_name = data.categories_names[c].split("_")[0]

        # NaN happens when there are only 0s, which might happen for very rare words or
        # very insufficient word vocabulary
        # ---

        cor = scipy.stats.spearmanr(scores, data.y[c]).correlation

        results[c_name].append(0 if np.isnan(cor) else cor)

    final_results = OrderedDict()

    final_results['all'] = sum(sum(v) for v in results.values()) / len(categories)

    for k in results:

        final_results[k] = sum(results[k]) / len(results[k])

    series = pd.Series(final_results)

    df = series.to_frame(name='accuracy')

    df['category'] = df.index

    df['dataset'] = 'SemEval'

    df['task'] = 'analogy'

    return df


def evaluate_on_WordRep(w, max_pairs=1000, solver_kwargs={}):
    """
    Evaluate on WordRep dataset

    Parameters
    ----------
    w : Embedding or dict
      Embedding or dict instance.

    max_pairs: int, default: 1000
      Each category will be constrained to maximum of max_pairs pairs
      (which results in max_pair * (max_pairs - 1) examples)

    solver_kwargs: dict, default: {}
      Arguments passed to SimpleAnalogySolver. It is suggested to limit number of words
      in the dictionary.

    References
    ----------
    Bin Gao, Jiang Bian, Tie-Yan Liu (2015)
     "WordRep: A Benchmark for Research on Learning Word Representations"
    """
    if isinstance(w, dict):

        w = Embedding.from_dict(w)

    data = fetch_wordrep()

    categories = set(data.category)

    accuracy = {}

    correct = {}

    count = {}

    missing = {}

    for category in categories:

        X_cat = data.X[data.category == category]

        # further limit the number of pairs to consider
        # ---
        X_cat = X_cat[0:max_pairs]

        nb_pairs = X_cat.shape[0]

        nb_questions = nb_pairs * (nb_pairs - 1)

        logger.info("Processing {} with {} pairs, {} questions".
                    format(category, nb_pairs, nb_questions))

        # For each category construct question-answer pairs
        # ---

        X = np.zeros(shape=(nb_questions, 3), dtype="object")

        y = np.zeros(shape=(nb_questions,), dtype="object")

        id = 0

        # to find all permutations
        # iterate through the Cartesian product
        # ---

        for left, right in product(X_cat, X_cat):

            if not np.array_equal(left, right):

                # we exclude the cases when left = right
                # ---

                X[id, 0:2] = left

                X[id, 2] = right[0]

                y[id] = right[1]

                id += 1

        # Run solver
        # ---

        solver = SimpleAnalogySolver(w=w, **solver_kwargs)

        results = solver.predict(X)

        y_pred = results['predictions']

        nb_correct = float(np.sum(y_pred == y))

        correct[category] = nb_correct

        count[category] = nb_questions

        accuracy[category] = nb_correct / nb_questions

        missing[category] = results['missing_words']

    # Add summary results
    # ---

    missing['wikipedia'] = sum(missing[c] for c in categories if c in data.wikipedia_categories)

    missing['all'] = sum(missing[c] for c in categories)

    missing['wordnet'] = sum(missing[c] for c in categories if c in data.wordnet_categories)

    correct['wikipedia'] = sum(correct[c] for c in categories if c in data.wikipedia_categories)

    correct['all'] = sum(correct[c] for c in categories)

    correct['wordnet'] = sum(correct[c] for c in categories if c in data.wordnet_categories)

    count['wikipedia'] = sum(count[c] for c in categories if c in data.wikipedia_categories)

    count['all'] = sum(count[c] for c in categories)

    count['wordnet'] = sum(count[c] for c in categories if c in data.wordnet_categories)

    accuracy['wikipedia'] = correct['wikipedia'] / count['wikipedia']

    accuracy['all'] = correct['all'] / count['all']

    accuracy['wordnet'] = correct['wordnet'] / count['wordnet']

    data = [pd.Series(accuracy, name="accuracy"),
            pd.Series(correct, name="correct"),
            pd.Series(count, name="count"),
            pd.Series(missing, name="missing")]

    df = pd.concat(data, axis=1)

    df['category'] = df.index

    df['dataset'] = 'WordRep'

    df['task'] = 'analogy'

    return df


def evaluate_on_BATS(w, solver_kwargs={}):
    """
    Evaluate on the BATS dataset

    Parameters
    ----------
    w : Embedding or dict
      Embedding or dict instance.

    solver_kwargs: dict, default: {}
      Arguments passed to SimpleAnalogySolver. It is suggested to limit number of words
      in the dictionary.

    References
    ----------
    Gladkova, A., Drozd, A., & Matsuoka, S. (2016). Analogy-Based Detection of Morphological and Semantic Relations with Word Embeddings: What Works and What Doesn’t. In Proceedings of the NAACL-HLT SRW (pp. 47–54). San Diego, California, June 12-17, 2016: ACL. https://doi.org/10.18653/v1/N16-2002

    """
    if isinstance(w, dict):

        w = Embedding.from_dict(w)

    data = fetch_BATS()

    categories = set(data.category)

    # just used two categories --- for testing purposes
    # categories = list(categories)[0:2]

    accuracy = {}

    correct = {}

    count = {}

    missing_words = {}

    for category in categories:

        pairs = data.X[data.category == category]

        # convert numpy array to list of lists
        # ---
        pairs = pairs.tolist()

        # we want to keep only the pairs covered
        # ---

        # filter 1
        # ---
        pairs = [(target_word, candidate) for target_word, candidate in pairs if target_word in w]

        # filter 2
        # ---
        final_pairs = []

        for target_word, candidate in pairs:

            found_word = False

            if "/" not in candidate:

                if candidate in w:

                    found_word = True

            else:

                words = candidate.split("/")

                for word in words:

                    if word in w:

                        # keep as a candidate the first word
                        # found in the vocabulary
                        # ---

                        found_word = True

                        candidate = word

                        break

            if found_word:

                word_pair = (target_word, candidate)

                final_pairs.append(word_pair)

        nb_pairs = len(final_pairs)

        if nb_pairs == 0:

            continue

        nb_questions = nb_pairs * (nb_pairs - 1)

        logger.info("Processing {} with {} pairs, {} questions".
                    format(category, nb_pairs, nb_questions))

        # For each category construct question-answer pairs
        # ---

        X = np.zeros(shape=(nb_questions, 3), dtype="object")

        y = np.zeros(shape=(nb_questions,), dtype="object")

        id = 0

        # to find all permutations
        # iterate through the Cartesian product
        # ---

        for left, right in product(final_pairs, final_pairs):

            if not np.array_equal(left, right):

                # we exclude the cases when left = right
                # ---

                X[id, 0:2] = left

                X[id, 2] = right[0]

                y[id] = right[1]

                id += 1

        # Run solver
        # ---

        solver = SimpleAnalogySolver(w=w, **solver_kwargs)

        results = solver.predict(X)

        nb_correct = float(np.sum(results['predictions'] == y))

        correct[category] = nb_correct

        count[category] = nb_questions

        accuracy[category] = nb_correct / nb_questions

        missing_words[category] = results['missing_words']

    # Add summary results
    # ---

    data = [pd.Series(accuracy, name="accuracy"),
            pd.Series(correct, name="correct"),
            pd.Series(count, name="count"),
            pd.Series(missing_words, name="missing")]

    return pd.concat(data, axis=1)


def cosine_similarity(vector1, vector2):
    '''
    angular difference between two vectors
    =
    ratio of the dot product and the product of the magnitudes of vectors
    v1-v2 / |v1||v2|
    '''

    dot_product = np.dot(vector1, vector2)

    vector1_mag = np.linalg.norm(vector1)

    vector2_mag = np.linalg.norm(vector2)

    result = dot_product / (vector1_mag * vector2_mag)

    return result


def answer_SAT_analogy_question(question, answers, w, solver):
    '''

    '''

    nb_rows = len(answers)

    # init
    # ---
    X = np.zeros(shape=(nb_rows, 3), dtype="object")

    y = np.zeros(shape=(nb_rows,), dtype="object")

    # for i in range(nb_rows):

    #     print("triple", i + 1, ":", X[i, ], "candidate:", y[i])

    # filling
    # ---

    for i, answer in enumerate(answers):

        X[i, 0] = question[0]
        X[i, 1] = question[1]
        X[i, 2] = answer[0]
        y[i] = answer[1]

    # for i in range(nb_rows):

    #     print("triple", i + 1, ":", X[i, ], "candidate:", y[i])

    # prediction through the analogy solver
    # ---

    results = solver.predict(X)

    y_pred = results['predictions']
    missing_words = results['missing_words']

    selected_answer = None
    selected_cosine = None

    for i in range(nb_rows):

        # prediction
        # ---

        predicted_word = y_pred[i]

        predicted_vector = w[predicted_word]

        # candidate
        # ---

        candidate_word = y[i]

        if candidate_word in w:

            candidate_vector = w[candidate_word]

            cosine = cosine_similarity(predicted_vector, candidate_vector)

            if selected_answer is None or cosine >= selected_cosine:

                selected_answer = i
                selected_cosine = cosine

        else:

            # print("The candidate word is not in the vocabulary. This item is ignored.")

            pass

        # print("triple", i + 1, ":", X[i, ], ", candidate:", candidate_word, ", prediction:", predicted_word, ", cosine:", cosine)

    # i = selected_answer

    # print("\nSelected answer: triple", i + 1, ":", X[i, ], ", candidate:", y[i])

    results = {'selected_answer': selected_answer, 'missing_words': missing_words}

    return results


def evaluate_on_SAT(w, solver_kwargs={}):
    """
    Evaluate on the SAT dataset

    Parameters
    ----------
    w : Embedding or dict
      Embedding or dict instance.

    solver_kwargs: dict, default: {}
      Arguments passed to SimpleAnalogySolver. It is suggested to limit number of words
      in the dictionary.

    References
    ----------
    Turney, P. D., Littman, M. L., Bigham, J., & Shnayder, V. (2003). Combining independent modules to solve multiple-choice synonym and analogy problems. In Proceedings of the International Conference on Recent Advances in Natural Language Processing (RANLP-03).

    """
    if isinstance(w, dict):

        w = Embedding.from_dict(w)

    solver = SimpleAnalogySolver(w=w, **solver_kwargs)
    # solver = SimpleAnalogySolver(w=w)

    data = fetch_SAT()

    nb_items = len(data.X)

    nb_items_correct = 0

    missing_words = 0

    for i in range(nb_items):

        question = data.X[i].split("_")

        answers = data.y[i, :]

        '''
            We split while testing for string to avoid attempting to split
            nan values, which occurs when there are 4 alternatives instead of 5.
        '''

        answers = [answer.split("_") for answer in answers
                   if isinstance(answer, six.string_types)]

        # print(question, answers)

        response = answer_SAT_analogy_question(question, answers, w, solver)

        missing_words += response['missing_words']

        i = response['selected_answer']

        # print(i)

        if i == 0:
            # this is the good answer
            nb_items_correct += 1
            # print("\n*** Yes! ***")

    # print("\nNumber of items:", nb_items, "Number of correct answers:", nb_items_correct)

    accuracy = nb_items_correct / nb_items

    data = [pd.Series(accuracy, name="accuracy"),
            pd.Series(nb_items_correct, name="correct"),
            pd.Series(nb_items, name="count"),
            pd.Series(missing_words, name="missing")]

    df = pd.concat(data, axis=1)

    df['dataset'] = 'SAT'

    df['task'] = 'analogy'

    return df


def answer_synonymy_question(question, answers, w):
    '''

    '''

    if question in w:

        question_vector = w[question]

    else:
        # if we do not have a vector for the question
        # we cannot answer it
        # ---

        response = {'selected_answer': None, 'selected_cosine': None}

        return response

    selected_answer = None

    selected_cosine = None

    nb_answers = len(answers)

    # chose the answer which has the highest cosine
    # ---

    for i in range(nb_answers):

        answer = answers[i]

        if answer in w:

            answer_vector = w[answer]

            cosine = cosine_similarity(question_vector, answer_vector)

            if selected_answer is None or cosine >= selected_cosine:

                selected_answer = i

                selected_cosine = cosine

    response = {'selected_answer': selected_answer, 'selected_cosine': selected_cosine}

    return response


def evaluate_on_synonyms(w, dataset_name):
    '''

    '''

    if isinstance(w, dict):

        w = Embedding.from_dict(w)

    # set the fetch function name
    # ---
    fetch_function_name = "fetch_" + dataset_name

    # retrieve the dataset
    # ---
    data = getattr(web.datasets.synonymy, fetch_function_name)()

    # the question
    # ---
    X = data.X

    # the answers
    # ---
    y = data.y

    nb_items = data.X.shape[0]

    nb_items_correct = 0

    missing = 0

    for i in range(nb_items):

        question = X[i]

        answers = y[i]

        # print(question, answers)

        response = answer_synonymy_question(question, answers, w)

        selected_answer = response['selected_answer']
        selected_cosine = response['selected_cosine']

        # print(i)

        if selected_answer is None:

            # no answer was given because the word was not in the vocabulary
            # ---

            missing += 1

        elif selected_answer == 0:
            # this is the good answer
            # ---
            nb_items_correct += 1

    accuracy = nb_items_correct / nb_items

    data = [pd.Series(accuracy, name="accuracy"),
            pd.Series(nb_items_correct, name="correct"),
            pd.Series(nb_items, name="count"),
            pd.Series(missing, name="missing")]

    results = pd.concat(data, axis=1)

    return results


if __name__ == "__main__":

    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')

    logger = logging.getLogger(__name__)

    w = load_toy_embedding()

    results = evaluate_on_all(w)

    print("\n\nFINAL RESULTS\n---\n")
    print(results)
