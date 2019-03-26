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

logger = logging.getLogger(__name__)


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

    return

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

    analogy_results["SemEval2012_2"] = evaluate_on_semeval_2012_2(w)['all']

    logger.info("Analogy prediction accuracy on {} {}".format("SemEval2012", analogy_results["SemEval2012_2"]))

    # Calculate results on categorization
    # ---

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
    # ---

    for name, data in iteritems(categorization_tasks):

        categorization_results[name] = evaluate_categorization(w, data.X, data.y)

        logger.info("Cluster purity on {} {}".format(name, categorization_results[name]))

    # Construct pd table

    cat = pd.DataFrame([categorization_results])

    analogy = pd.DataFrame([analogy_results])

    sim = pd.DataFrame([similarity_results])

    results = cat.join(sim).join(analogy)

    return results


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

    analogy_results["SemEval2012_2"] = evaluate_on_semeval_2012_2(w)['all']

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

    results = solver.predict(X)

    y_pred = results['predictions']

    if category is not None:

        results = OrderedDict({"all": np.mean(y_pred == y)})

        count = OrderedDict({"all": len(y_pred)})

        correct = OrderedDict({"all": np.sum(y_pred == y)})

        for cat in set(category):

            results[cat] = np.mean(y_pred[category == cat] == y[category == cat])

            count[cat] = np.sum(category == cat)

            correct[cat] = np.sum(y_pred[category == cat] == y[category == cat])

        return pd.concat([pd.Series(results, name="accuracy"),
                          pd.Series(correct, name="correct"),
                          pd.Series(count, name="count")],
                         axis=1)

    else:

        return np.mean(y_pred == y)


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

    return pd.Series(final_results)


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

    results = pd.concat(data, axis=1)

    return results


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

    results = pd.concat(data, axis=1)

    return results


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


def quick_test():

    # from web.embeddings import fetch_GloVe
    # w = fetch_GloVe(corpus="wiki-6B", dim=50)

    # let's load a smaller file
    # ---

    from web.embeddings import load_embedding

    input_path = '~/web_data/embeddings/glove.6B/glove.6B.50d.SAMPLE.txt'
    fname = os.path.expanduser(input_path)

    load_kwargs = {"vocab_size": 50000, "dim": 50}

    w = load_embedding(fname=fname,
                       format="glove",
                       normalize=True,
                       lower=False,
                       clean_words=False,
                       load_kwargs=load_kwargs)

    datasets = []
    datasets.append("WordRep")
    # datasets.append("SimVerb3500")
    # datasets.append("battig2010")
    # datasets.append("BATS")
    # datasets.append("SAT")
    # datasets.append("TOEFL")
    # datasets.append("ESL")

    # SimVerb3500 : similarity / Spearman correlation
    # ---

    if 'SimVerb3500' in datasets:

        data = fetch_SimVerb3500()

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

    quick_test()
