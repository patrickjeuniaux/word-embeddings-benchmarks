"""
This file only contains 'Embedding',
the base class for word embeddings.

NOTE: This file was adapted from the polyglot package

Polyglot is a natural language pipeline that supports
massive multilingual applications.
https://polyglot.readthedocs.io/en/latest/index.html

"""

# external imports
# ---

import logging
from collections import OrderedDict

import numpy as np
import sys

from functools import partial
from sklearn.metrics import pairwise_distances

from six import text_type
from six import PY2
from six import iteritems
from six import string_types
from six.moves import cPickle as pickle
from six.moves import range

# internal imports
# ---

from .utils import _open
from .utils import standardize_string
from .utils import to_utf8
from .vocabulary import Vocabulary
from .vocabulary import CountedVocabulary
from .vocabulary import OrderedVocabulary


logger = logging.getLogger(__name__)


class Embedding(object):
    """
        Mapping a n-word vocabulary to n d-dimensional points.
    """

    def __init__(self, vocabulary, vectors):
        """
            Vocabulary : a Vocabulary object
                         containing a {word:ID} dict
                         and its reversed mapping, a {ID:word} dict

            Vectors : a list of list of numbers
                        where each list of numbers
                        stands for a word's vector
        """

        self.vocabulary = vocabulary

        self.vectors = np.asarray(vectors)

        if len(self.vocabulary) != self.vectors.shape[0]:

            raise ValueError("Vocabulary has {} items but we have {} "
                             "vectors."
                             .format(len(vocabulary), self.vectors.shape[0]))

        if len(self.vocabulary.words) != len(set(self.vocabulary.words)):

            logger.warning("Vocabulary has duplicates.")

    def __getitem__(self, word):
        """
            Return the vector for the word in the vocabulary
        """

        return self.vectors[self.vocabulary[word]]

    def __setitem__(self, word, vector):
        """
            Assign the vector to the word in the vocabulary
        """

        if not vector.shape[0] == self.vectors.shape[1]:

            raise RuntimeError("Please pass vector of len {}".format(self.vectors.shape[1]))

        if word not in self.vocabulary:

            # If the word is not in the vocabulary,
            # add it !

            self.vocabulary.add(word)

            # and stack the new vector on top of those that are already there

            self.vectors = np.vstack([self.vectors, vector.reshape(1, -1)])
        else:

            # the word already exists
            # replace its vector by the new vector

            self.vectors[self.vocabulary[word]] = vector

    def __contains__(self, word):
        """
            Check whether the word is in the vocabulary
        """

        return word in self.vocabulary

    def __delitem__(self, word):
        """
            Remove the word and its vector from the embedding.

            Note:
             This operation costs \\theta(n). Be careful putting it in a loop.
        """
        id = self.vocabulary[word]

        del self.vocabulary[word]

        self.vectors = np.delete(self.vectors, id, 0)

    def __len__(self):
        """
            Return the number of words in the vocabulary
        """
        return len(self.vocabulary)

    def __iter__(self):
        """
            Iterate over the words in the vocabulary

            yielding (word, vector) tuples
        """
        for word in self.vocabulary:

            yield word, self[word]

    @property
    def words(self):
        """
            Return the words in the vocabulary
        """
        return self.vocabulary.words

    @property
    def shape(self):
        """

        """
        return self.vectors.shape

    def get(self, word, default=None):
        """
            Get the vector of the word
            or a default value if the word does not exist

        """
        try:

            return self[word]

        except KeyError as e:

            return default

    def standardize_words(self, lower=False, clean_words=False, inplace=False):
        """

        """
        new_embedding = self.transform_words(partial(standardize_string, lower=lower, clean_words=clean_words),
                                             inplace=inplace,
                                             lower=lower)

        if clean_words:

            new_embedding = new_embedding.transform_words(partial(lambda w: w.strip(" ")),
                                                          inplace=inplace,
                                                          lower=lower)

        return new_embedding

    def transform_words(self, f, inplace=False, lower=False):
        """
            Transform words in vocabulary according to following strategy:
            Prefer shortest and most often occurring words - after transforming by some (lambda f) function.

            This allows the elimination of noisy and wrongly coded words.

            Strategy is implemented for all types of Vocabulary - they can be polymorphicaly extended.

            Parameters
            ----------
            f: lambda
                Function called on each word to transform it

            inplace: bool, default: False
                Return new Embedding instance or modify existing

            lower: bool, default: False
                If true, will convert all words to lowercase

            Returns
            -------
            e: Embedding
            Instance of Embedding class with the same Vocabulary type as previous.
        """

        id_map = OrderedDict()

        word_count = len(self.vectors)

        # store max word length before f(w)- in corpora
        words_len = {}

        # store max occurrence count of word
        counts = {}

        is_vocab_generic = False

        curr_words = self.vocabulary.words

        curr_vec = self.vectors

        if isinstance(self.vocabulary, CountedVocabulary):

            _, counter_of_words = self.vocabulary.getstate()

        elif isinstance(self.vocabulary, OrderedVocabulary):

            # range in python3 is lazy
            counter_of_words = range(len(self.vocabulary.words) - 1, -1, -1)

        elif isinstance(self.vocabulary, Vocabulary):

            is_vocab_generic = True

            # if corpora contain lowercase version of word i- for case Vocabulary
            lowered_words = {}

            if lower:

                for w, v in zip(self.vocabulary.words, self.vectors):

                    wl = w.lower()

                    if wl == w:

                        lowered_words[wl] = v

                    elif wl != w and wl not in lowered_words:

                        lowered_words[wl] = v

                curr_words = list(lowered_words.keys())

                curr_vec = np.asanyarray(list(lowered_words.values()))

        else:
            raise NotImplementedError(
                'This kind of Vocabulary is not implemented in transform_words strategy and can not be matched')

        for id, w in enumerate(curr_words):

            fw = f(w)

            if len(fw) and fw not in id_map:

                id_map[fw] = id

                if not is_vocab_generic:

                    counts[fw] = counter_of_words[id]

                words_len[fw] = len(w)

                # overwrite

            elif len(fw) and fw in id_map:

                if not is_vocab_generic and counter_of_words[id] > counts[fw]:

                    id_map[fw] = id

                    counts[fw] = counter_of_words[id]

                    words_len[fw] = len(w)

                elif is_vocab_generic and len(w) < words_len[fw]:

                    # for generic Vocabulary
                    id_map[fw] = id

                    words_len[fw] = len(w)

                elif not is_vocab_generic and counter_of_words[id] == counts[fw] and len(w) < words_len[fw]:

                    id_map[fw] = id

                    counts[fw] = counter_of_words[id]

                    words_len[fw] = len(w)

                logger.warning("Overwriting {}".format(fw))

        if isinstance(self.vocabulary, CountedVocabulary):

            words_only = id_map.keys()

            vectors = curr_vec[[id_map[w] for w in words_only]]

            words = {w: counter_of_words[id_map[w]] for w in words_only}

        elif isinstance(self.vocabulary, OrderedVocabulary):

            words = sorted(id_map.keys(), key=lambda x: id_map[x])

            vectors = curr_vec[[id_map[w] for w in words]]

        elif isinstance(self.vocabulary, Vocabulary):

            words = sorted(id_map.keys(), key=lambda x: id_map[x])

            vectors = curr_vec[[id_map[w] for w in words]]

        logger.info("Transformed {} into {} words".format(word_count, len(words)))

        if inplace:

            self.vectors = vectors

            self.vocabulary = self.vocabulary.__class__(words)

            return self

        else:
            return Embedding(vectors=vectors, vocabulary=self.vocabulary.__class__(words))

    def most_frequent(self, k, inplace=False):
        """
            Only most frequent k words to be included in the embeddings.
        """

        assert isinstance(self.vocabulary, OrderedVocabulary), \
            "most_frequent can be called only on Embedding with OrderedVocabulary"

        vocabulary = self.vocabulary.most_frequent(k)

        vectors = np.asarray([self[w] for w in vocabulary])

        if inplace:

            self.vocabulary = vocabulary

            self.vectors = vectors

            return self

        return Embedding(vectors=vectors, vocabulary=vocabulary)

    def normalize_words(self, ord=2, inplace=False):
        """
            Normalize embeddings matrix row-wise.

            Parameters
            ----------
              ord: normalization order. Possible values {1, 2, 'inf', '-inf'}
        """
        if ord == 2:

            ord = None  # numpy uses this flag to indicate l2.

        vectors = self.vectors.T / np.linalg.norm(self.vectors, ord, axis=1)

        if inplace:

            self.vectors = vectors.T

            return self

        return Embedding(vectors=vectors.T, vocabulary=self.vocabulary)

    def nearest_neighbors(self, word, k=1, exclude=[], metric="cosine"):
        """
            Find nearest neighbor of given word

            Parameters
            ----------
              word: string or vector
                Query word or vector.

              k: int, default: 1
                Number of nearest neighbours to return.

              metric: string, default: 'cosine'
                Metric to use.

              exclude: list, default: []
                Words to omit in answer

            Returns
            -------
              n: list
                Nearest neighbors.
        """
        if isinstance(word, string_types):

            assert word in self, "Word not found in the vocabulary"

            vector = self[word]

        else:

            vector = word

        D = pairwise_distances(self.vectors, vector.reshape(1, -1), metric=metric)

        if isinstance(word, string_types):

            D[self.vocabulary.word_id[word]] = D.max()

        for excluded_word in exclude:

            D[self.vocabulary.word_id[excluded_word]] = D.max()

        return [self.vocabulary.id_word[id] for id in D.argsort(axis=0).flatten()[0:k]]

    @staticmethod
    def from_gensim(model):
        """

        """

        word_count = {}

        vectors = []

        for word, vocab in sorted(iteritems(model.vocab), key=lambda item: -item[1].count):

            word = standardize_string(word)

            if word:

                vectors.append(model.syn0[vocab.index])

                word_count[word] = vocab.count

        vocab = CountedVocabulary(word_count=word_count)

        vectors = np.asarray(vectors)

        return Embedding(vocabulary=vocab, vectors=vectors)

    @staticmethod
    def from_word2vec_vocab(fvocab):
        """

        """

        counts = {}

        with _open(fvocab) as fin:

            for line in fin:

                word, count = standardize_string(line).split()

                if word:

                    counts[word] = int(count)

        return CountedVocabulary(word_count=counts)

    @staticmethod
    def _from_word2vec_binary(fname):
        """

        """
        with _open(fname, 'rb') as fin:

            words = []

            header = fin.readline()

            vocab_size, layer1_size = list(map(int, header.split()))  # throws for invalid file format

            logger.info("Loading #{} words with {} dim".format(vocab_size, layer1_size))

            vectors = np.zeros((vocab_size, layer1_size), dtype=np.float32)

            binary_len = np.dtype("float32").itemsize * layer1_size

            for line_no in range(vocab_size):

                # mixed text and binary: read text first, then binary
                word = []

                while True:

                    ch = fin.read(1)

                    if ch == b' ':

                        break

                    if ch != b'\n':  # ignore newlines in front of words (some binary files have newline, some don't)

                        word.append(ch)

                words.append(b''.join(word).decode("latin-1"))

                vectors[line_no, :] = np.fromstring(fin.read(binary_len), dtype=np.float32)

            if len(words) < vocab_size:

                logger.warning("Omitted {} words".format(vocab_size - len(words)))

            elif len(words) > vocab_size:

                raise RuntimeError("Read too many words, incorrect file")

            return words, vectors

    @staticmethod
    def _from_word2vec_text(fname):
        """

        """
        with _open(fname, 'r') as fin:

            words = []

            header = fin.readline()

            ignored = 0

            vocab_size, layer1_size = list(map(int, header.split()))  # throws for invalid file format

            vectors = np.zeros(shape=(vocab_size, layer1_size), dtype=np.float32)

            for line_no, line in enumerate(fin):

                try:

                    parts = text_type(line, encoding="utf-8").split(' ')

                    w = parts[0]

                    parts = list(map(lambda x: x.strip(), parts[1:]))

                    parts.insert(0, w)

                except TypeError as e:

                    parts = line.split(' ')

                    w = parts[0]

                    parts = list(map(lambda x: x.strip(), parts[1:]))

                    parts.insert(0, w)

                except Exception as e:

                    logger.warning("We ignored line number {} because of errors in parsing"
                                   "\n{}".format(line_no, e))

                    continue

                # We differ from Gensim implementation.
                # Our assumption that a difference of one happens because of having a
                # space in the word.

                if len(parts) == layer1_size + 1:

                    word, vectors[line_no - ignored] = parts[0], list(map(np.float32, parts[1:]))

                elif len(parts) == layer1_size + 2 and parts[-1]:

                    # last element after splitting is not empty- some glove corpora have additional space
                    word, vectors[line_no - ignored] = parts[:2], list(map(np.float32, parts[2:]))

                    word = u" ".join(word)

                elif not parts[-1]:

                    # omit last value - empty string
                    word, vectors[line_no - ignored] = parts[0], list(map(np.float32, parts[1:-1]))

                else:

                    ignored += 1

                    logger.warning("We ignored line number {} because of unrecognized "
                                   "number of columns {}".format(line_no, parts[:-layer1_size]))

                    continue

                words.append(word)

            if ignored:

                vectors = vectors[0:-ignored]

            if len(words) < vocab_size:

                logger.warning("Omitted {} words".format(vocab_size - len(words)))

            elif len(words) > vocab_size:

                raise RuntimeError("Read too many words, incorrect file")

            return words, vectors

    @staticmethod
    def from_glove(fname, vocab_size, dim):
        """

        """
        with _open(fname, 'r') as fin:

            words = []

            words_uniq = set()

            ignored = 0

            vectors = np.zeros(shape=(vocab_size, dim), dtype=np.float32)

            for line_no, line in enumerate(fin):

                try:

                    parts = text_type(line, encoding="utf-8").split(' ')

                    parts[1:] = map(lambda x: np.float32(x.strip()), parts[1:])

                except TypeError as e:

                    parts = line.split(' ')

                    parts[1:] = map(lambda x: np.float32(x.strip()), parts[1:])

                except Exception as e:

                    ignored += 1

                    logger.warning("We ignored line number {} because of errors in parsing"
                                   "\n{}".format(line_no, e))

                    continue

                try:

                    if parts[0] not in words_uniq:

                        word, vectors[line_no - ignored] = parts[0], list(parts[len(parts) - dim:])

                        words.append(word)

                        words_uniq.add(word)

                    else:

                        ignored += 1

                        logger.warning(
                            "We ignored line number {} - following word is duplicated in file:\n{}\n".format(line_no,
                                                                                                             parts[0]))

                except Exception as e:

                    ignored += 1

                    logger.warning("We ignored line number {} because of errors in parsing"
                                   "\n{}".format(line_no, e))

            return Embedding(vocabulary=OrderedVocabulary(words), vectors=vectors[0:len(words)])

    @staticmethod
    def from_dict(d):
        """

        """
        for k in d:  # Standardize

            d[k] = np.array(d[k]).flatten()

        return Embedding(vectors=list(d.values()), vocabulary=Vocabulary(d.keys()))

    @staticmethod
    def to_word2vec(w, fname, binary=False):
        """
            Store the input-hidden weight matrix in the same format used by the original
            C word2vec-tool, for compatibility.

            Parameters
            ----------
            w: Embedding instance

            fname: string
              Destination file
        """

        logger.info("storing %sx%s projection weights into %s" % (w.vectors.shape[0], w.vectors.shape[1], fname))

        with _open(fname, 'wb') as fout:

            fout.write(to_utf8("%s %s\n" % w.vectors.shape))

            # store in sorted order: most frequent words at the top
            for word, vector in zip(w.vocabulary.words, w.vectors):

                if binary:

                    fout.write(to_utf8(word) + b" " + vector.astype("float32").tostring())

                else:

                    fout.write(to_utf8("%s %s\n" % (word, ' '.join("%.15f" % val for val in vector))))

    @staticmethod
    def from_word2vec(fname, fvocab=None, binary=False):
        """
            Load the input-hidden weight matrix from the original C word2vec-tool format.

            `binary` is a boolean indicating whether the data is in binary word2vec format.
            Word counts are read from `fvocab` filename, if set (this is the file generated
            by `-save-vocab` flag of the original C tool).
        """

        vocabulary = None

        if fvocab is not None:

            logger.info("loading word counts from %s" % (fvocab))

            vocabulary = Embedding.from_word2vec_vocab(fvocab)

        logger.info("loading projection weights from %s" % (fname))

        if binary:

            words, vectors = Embedding._from_word2vec_binary(fname)

        else:

            words, vectors = Embedding._from_word2vec_text(fname)

        if not vocabulary:

            vocabulary = OrderedVocabulary(words=words)

        if len(words) != len(set(words)):

            raise RuntimeError("Vocabulary has duplicates")

        e = Embedding(vocabulary=vocabulary, vectors=vectors)

        return e

    @staticmethod
    def load(fname):
        """
            Load an embedding dump generated by `save`
        """

        content = _open(fname).read()

        if PY2:

            state = pickle.loads(content, encoding='latin1')

        else:

            state = pickle.loads(content, encoding='latin1')

        voc, vec = state

        if len(voc) == 2:

            words, counts = voc

            word_count = dict(zip(words, counts))

            vocab = CountedVocabulary(word_count=word_count)

        else:

            vocab = OrderedVocabulary(voc)

        return Embedding(vocabulary=vocab, vectors=vec)

    def save(self, fname):
        """
            Save a pickled version of the embedding into `fname`.
        """

        vec = self.vectors

        voc = self.vocabulary.getstate()

        state = (voc, vec)

        with open(fname, 'wb') as f:

            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
