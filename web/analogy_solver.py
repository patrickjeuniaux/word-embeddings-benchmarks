"""
 Classes and function for answering analogy questions
"""

# external imports
# ---

import logging
from collections import OrderedDict
import six
from six.moves import range
import scipy
import pandas as pd
from itertools import product
import sklearn

logger = logging.getLogger(__name__)

# internal imports
# ---

from web.datasets.analogy import *
from web.utils import batched
from web.embedding import Embedding


class SimpleAnalogySolver(sklearn.base.BaseEstimator):
    """
    Answer analogy questions

    Parameters
    ----------
    w : Embedding instance

    method : {"add", "mul"}
      Method to use when finding analogy answer, see "Improving Distributional Similarity
      with Lessons Learned from Word Embeddings" O. Levy et al. 2014.

    batch_size : int
      Batch size to use while computing accuracy. This is because of extensive memory usage.

    k: int
      If not None will select k top most frequent words from embedding before doing analogy prediction
      (this can offer significant speedups)

    Note
    ----
    It is suggested to normalize and standardize embedding before passing it to SimpleAnalogySolver.
    To speed up code consider installing OpenBLAS and setting OMP_NUM_THREADS.
    """

    def __init__(self, w, method="add", batch_size=300, k=None):

        self.w = w

        self.batch_size = batch_size

        self.method = method

        self.k = k

    def score(self, X, y):
        """
        Calculate accuracy on analogy questions dataset

        Parameters
        ----------
        X : array-like, shape (n_samples, 3)
          Analogy questions.

        y : array-like, shape (n_samples, )
          Analogy answers.

        Returns
        -------
        acc : float
          Accuracy
        """
        return np.mean(y == self.predict(X))

    def predict(self, X):
        """
        Answer analogy questions

        Parameters
        ----------
        X : array-like, shape (n_samples, 3)
          Analogy questions.

        Returns
        -------
        y_pred : array-like, shape (n_samples, )
          Predicted words.
        """
        w = self.w.most_frequent(self.k) if self.k else self.w

        words = self.w.vocabulary.words

        word_id = self.w.vocabulary.word_id

        mean_vector = np.mean(w.vectors, axis=0)

        output = []

        missing_words = 0

        for query in X:

            for query_word in query:

                if query_word not in word_id:

                    missing_words += 1

        if missing_words > 0:

            logger.warning("Missing {} words. Will replace them with mean vector".format(missing_words))

        # Batch due to memory constraints (in dot operation)

        for id_batch, batch in enumerate(batched(range(len(X)), self.batch_size)):

            ids = list(batch)

            X_b = X[ids]

            if id_batch % np.floor(len(X) / (10. * self.batch_size)) == 0:

                logger.info("Processing {}/{} batch".format(int(np.ceil(ids[1] / float(self.batch_size))),
                                                            int(np.ceil(X.shape[0] / float(self.batch_size)))))

            A, B, C = np.vstack(w.get(word, mean_vector) for word in X_b[:, 0]), \
                np.vstack(w.get(word, mean_vector) for word in X_b[:, 1]), \
                np.vstack(w.get(word, mean_vector) for word in X_b[:, 2])

            if self.method == "add":

                D = np.dot(w.vectors, (B - A + C).T)

            elif self.method == "mul":

                D_A = np.log((1.0 + np.dot(w.vectors, A.T)) / 2.0 + 1e-5)

                D_B = np.log((1.0 + np.dot(w.vectors, B.T)) / 2.0 + 1e-5)

                D_C = np.log((1.0 + np.dot(w.vectors, C.T)) / 2.0 + 1e-5)

                D = D_B - D_A + D_C

            else:

                raise RuntimeError("Unrecognized method parameter")

            # Remove words that were originally in the query

            for id, row in enumerate(X_b):

                D[[w.vocabulary.word_id[r] for r in row if r in

                   w.vocabulary.word_id], id] = np.finfo(np.float32).min

            output.append([words[id] for id in D.argmax(axis=0)])

        return np.array([item for sublist in output for item in sublist])


if __name__ == "__main__":

    import numpy as np
    from web.embeddings import fetch_GloVe
    from web.evaluate import cosine_similarity

    print("\nTest of 'SimpleAnalogySolver'")
    print("---")

    print("\nCreation of the data:")

    question = ('ostrich', 'bird')
    good_answer = ('lion', 'cat')
    bad_answers = [('goose', 'flock'), ('ewe', 'sheep'), ('cub', 'bear'), ('primate', 'monkey')]

    nb_rows = len(bad_answers) + 1

    # init
    # ---
    X = np.zeros(shape=(nb_rows, 3), dtype="object")

    y = np.zeros(shape=(nb_rows,), dtype="object")

    for i in range(nb_rows):

        print("triple", i + 1, ":", X[i, ], "candidate:", y[i])

    # filling
    # ---
    X[0, 0] = question[0]
    X[0, 1] = question[1]
    X[0, 2] = good_answer[0]
    y[0] = good_answer[1]

    for i, bad_answer in enumerate(bad_answers, 1):

        X[i, 0] = question[0]
        X[i, 1] = question[1]
        X[i, 2] = bad_answer[0]
        y[i] = bad_answer[1]

    for i in range(nb_rows):

        print("triple", i + 1, ":", X[i, ], "candidate:", y[i])

    print("\nLoad embeddings")
    print("Warning: it might take a few minutes")
    print("---")

    w = fetch_GloVe(corpus="wiki-6B", dim=300)

    print("\nPredictions via 'SimpleAnalogySolver':")

    # solver = SimpleAnalogySolver(w=w, **solver_kwargs)
    solver = SimpleAnalogySolver(w=w)

    y_pred = solver.predict(X)

    selected_answer = None
    selected_cosine = None

    for i in range(nb_rows):

        # prediction

        predicted_word = y_pred[i]

        predicted_vector = w[predicted_word]

        # candidate

        candidate_word = y[i]

        if candidate_word in w:

            candidate_vector = w[candidate_word]

            cosine = cosine_similarity(predicted_vector, candidate_vector)

            if selected_answer is None or cosine >= selected_cosine:

                selected_answer = i
                selected_cosine = cosine

        else:

            print("The candidate word is not in the vocabulary. This item is ignored.")

            cosine = None

        print("triple", i + 1, ":", X[i, ], ", candidate:", candidate_word, ", prediction:", predicted_word, ", cosine:", cosine)

    i = selected_answer

    print("\nSelected answer: triple", i + 1, ":", X[i, ], ", candidate:", y[i])

    print("")
    print("---THE END---")
