#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

This file contains :

- def count
- class Vocabulary
- class OrderedVocabulary (a subclass of Vocabulary)
- class CountedVocabulary (a subclass of OrderedVocabulary)

These objects support the 'embedding' module

NOTE: This file was adapted from the polyglot package

Polyglot is a natural language pipeline that supports
massive multilingual applications.
https://polyglot.readthedocs.io/en/latest/index.html

"""

# external imports
# ---

import os
from collections import Counter
from collections import OrderedDict
from io import open
from io import StringIO
from concurrent.futures import ProcessPoolExecutor

import six
from six import iteritems
from six import text_type as unicode
from six import string_types
from six.moves import zip

# internal imports
# ---

from web.utils import _open


def count(lines):
    """
       Counts the word frequencies in a list of sentences.

       This is a helper function for parallel execution of
       the `Vocabulary.from_text` method.
    """
    words = [word for line in lines for word in line.strip().split()]

    return Counter(words)


class Vocabulary(object):
    """
       A set of words/tokens that have consistent IDs.

       Attributes:

           word_id (dict): Mapping from words to IDs.

           id_word (dict): A reverse map of `word_id`.
    """

    def __init__(self, words=None):
        """
            Build attributes word_id and id_word from input.

            Args:

              words (list/set): list or set of words.
        """

        words = self.sanitize_words(words)

        self.word_id = {word: id for id, word in enumerate(words)}

        self.id_word = {id: word for word, id in iteritems(self.word_id)}

    def sanitize_words(self, words):
        """
            Guarantees that all textual symbols are unicode.

          Note:
              We do not convert numbers, only strings to unicode.
              We assume that the strings are encoded in utf-8.
        """
        _words = []

        for word in words:

            if isinstance(word, string_types) and not isinstance(word, unicode):

                _words.append(unicode(word, encoding="utf-8"))

            else:

                _words.append(word)

        return _words

    def __iter__(self):
        """
           Iterate over the words in a vocabulary

           in an ordered way, following the word's ID
        """

        for word, id in sorted(iteritems(self.word_id), key=lambda wc: wc[1]):

            yield word

    @property
    def words(self):
        """
           Return an ordered list of words according to their IDs.

           This uses __iter__
        """
        return list(self)

    def __unicode__(self):
        """
            Return a string with the words, one per line
        """

        return u"\n".join(self.words)

    def __str__(self):

        # if Python 3
        # ---
        if six.PY3:

            return self.__unicode__()

        # else (if Python 2...)
        # ---
        return self.__unicode__().encode("utf-8")

    def __getitem__(self, word):
        """
            Get the ID of a word
        """

        if isinstance(word, string_types) and not isinstance(word, unicode):

            word = unicode(word, encoding="utf-8")

        return self.word_id[word]

    def add(self, word):
        """
            Add a word to the vocabulary,

            specifying a new ID
        """

        if isinstance(word, string_types) and not isinstance(word, unicode):

            word = unicode(word, encoding="utf-8")

        if word in self.word_id:

            raise RuntimeError("Already existing word")

        id = len(self.word_id)

        self.word_id[word] = id

        self.id_word[id] = word

    def __contains__(self, word):
        """
            Check whether the word is already present in the vocabulary
        """

        return word in self.word_id

    def __delitem__(self, word):
        """
            Delete a word from vocabulary.

            Note:
             To maintain consecutive IDs, this operation was implemented
             with a complexity of \\theta(n).
        """
        del self.word_id[word]

        self.id_word = dict(enumerate(self.words))

        self.word_id = {w: id for id, w in iteritems(self.id_word)}

    def __len__(self):
        """
            Return the number of words in the vocabulary
        """

        return len(self.word_id)

    def get(self, word, default=None):
        """
            Return the ID of the word
            and a default value if it does not exist
        """

        try:

            return self[word]

        except KeyError as e:

            return default

    def getstate(self):
        """
            Return a list of words

            using the words property
        """

        return list(self.words)

    @classmethod
    def from_vocabfile(cls, filename):
        """
            Construct a CountedVocabulary out of a vocabulary file.

        Note:
          File has the following format :
                                        word1
                                        word2
        """
        words = [word.strip() for word in _open(filename, 'r').read().splitlines()]

        return cls(words=words)


class OrderedVocabulary(Vocabulary):
    """
        An ordered list of words/tokens according to their frequency.

        Note:
          The words are assumed to be sorted according to their frequency.
          Most frequent words appear first in the list.

        Attributes:
          word_id (dict): Mapping from words to IDs.
          id_word (dict): A reverse map of `word_id`.
    """

    def __init__(self, words=None):
        """
            Build attributes word_id and id_word from input.

            Args:
              words (list): list of sorted words according to frequency.
        """

        words = self.sanitize_words(words)

        self.word_id = {word: id for id, word in enumerate(words)}

        self.id_word = {id: word for word, id in iteritems(self.word_id)}

    def most_frequent(self, k):
        """
            Returns a vocabulary with the most frequent `k` words.

            Args:
              k (integer): specifies the top k most frequent words to be returned.
        """
        return OrderedVocabulary(words=self.words[:k])


class CountedVocabulary(OrderedVocabulary):
    """
        List of words and counts sorted according to word count.
    """

    def __init__(self, word_count=None):
        """

            Build attributes word_id and id_word from input.

            Args:
              word_count : A {word:count} dict
                           or
                           a list of (word, count) tuples.
        """

        if isinstance(word_count, dict):

            word_count = iteritems(word_count)

        sorted_counts = list(sorted(word_count, key=lambda wc: wc[1], reverse=True))

        words = [word for word, count in sorted_counts]

        super(CountedVocabulary, self).__init__(words=words)

        self.word_count = OrderedDict(sorted_counts)

    def most_frequent(self, k):
        """
            Returns a vocabulary with the most frequent `k` words.

            Args:
              k (integer): specifies the top k most frequent words to be returned.
        """

        word_count = [(word, self.word_count[word]) for word in self.words[:k]]

        return CountedVocabulary(word_count=word_count)

    def min_count(self, n=1):
        """
            Returns a vocabulary after eliminating the words that appear < `n`.

            Args:
              n (integer): specifies the minimum word frequency allowed.
        """

        word_count = [(word, count) for word, count in iteritems(self.word_count) if count >= n]

        return CountedVocabulary(word_count=word_count)

    def __unicode__(self):
        """

        """

        return u"\n".join([u"{}\t{}".format(word, self.word_count[w]) for word in self.words])

    def __delitem__(self, key):
        """

        """

        super(CountedVocabulary, self).__delitem__(key)

        self.word_count = OrderedDict([(word, self.word_count[word]) for word in self])

    def getstate(self):
        """

        """

        words = list(self.words)

        counts = [self.word_count[word] for word in words]

        return (words, counts)

    @staticmethod
    def from_vocabfile(filename):
        """
            Construct a CountedVocabulary out of a vocabulary file.

            Note:
              File has the following format :
                                            word1 count1
                                            word2 count2
        """
        word_count = [wc.strip().split() for wc in _open(filename, 'r').read().splitlines()]

        word_count = OrderedDict([(word, int(count)) for word, count in word_count])

        return CountedVocabulary(word_count=word_count)
