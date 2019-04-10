Main updates
============

This is a list of changes made to the original project.

Patrick Jeuniaux

University of Pisa


Dealing with NaN values
-----------------------

2019-04-08 --- 2019-04-10


In `web.embedding.Embedding <web/embedding.py>`_, added

    def remove_constant_vectors

in order to remove the vectors that are constants.

This removes vectors that contain no valuable information
and that may complicate the evaluation later on.

This was motivated by the existence of vectors containing only 0s
which would throw RuntimeWarning during normalization, i.e., when
attempting to perform:

    vectors = self.vectors.T / np.linalg.norm(self.vectors, ord, axis=1)



Standardization of the output
-----------------------------
2019-03-26 --- 2019-04-03

Most of the work was done in `web.evaluate <web/evaluate.py>`_.



New synonymy datasets 'TOEFL' and 'ESL'
---------------------------------------
2019-01-02 --- 2019-01-08

In `web.datasets.synonymy <web/datasets/synonymy.py>`_, added

    def fetch_TOEFL

    def fetch_ESL

This function loads the datasets from a local copy (not provided).

In `web.evaluate <web/evaluate.py>`_, added

    def answer_synonymy_question

    def evaluate_on_synonyms

This datasets are tested in `examples.test_datasets <examples/test_datasets.py>`_.





New analogy dataset 'SAT'
-------------------------
2019-01-01 --- 2019-01-08

In `web.datasets.analogy <web/datasets/analogy.py>`_, added

    def fetch_SAT

This function loads the dataset from a local copy (not provided).

In `web.evaluate <web/evaluate.py>`_, added

    def answer_SAT_analogy_question

    def evaluate_on_SAT

This dataset is tested in `examples.test_datasets <examples/test_datasets.py>`_.




New analogy dataset 'BATS'
--------------------------
2019-01-01 --- 2019-01-07

In `web.datasets.analogy <web/datasets/analogy.py>`_, added

    def fetch_BATS

This function loads the dataset from a local copy (not provided).

In `web.evaluate <web/evaluate.py>`_, added

    def evaluate_on_BATS

This dataset is tested in `examples.test_datasets <examples/test_datasets.py>`_.



New categorization dataset 'BATTIG2010'
---------------------------------------
2019-01-03 --- 2019-01-07

In `web.datasets.categorization <web/datasets/categorization.py>`_, added

    def fetch_battig2010

This function loads the dataset from a local copy (not provided).

This dataset is tested in `examples.test_datasets <examples/test_datasets.py>`_.






New similarity dataset 'SimVerb3500'
------------------------------------
2018-12-27 --- 2019-01-07

In `web.datasets.similarity <web/datasets/similarity.py>`_, added

    def fetch_SimVerb3500

This function downloads the dataset from the Internet.

In `web.evaluate <web/evaluate.py>`_

    def evaluate_on_all

added `fetch_SimVerb3500()`

This dataset is tested in `examples.test_new_datasets <examples/test_new_datasets.py>`_.



Calculate datasets statistics
-----------------------------
2018-12-31 --- 2019-01-06


Add `web.datasets.items.counter <web/datasets/items/counter.py>`_

to count the number of items in the datasets.

The use of this module is illustrated in

`scripts.count_items <scripts/count_items.py>`_


Add `web.datasets.items.coverage <web/datasets/items/coverage.py>`_

to determine the number of items in the datasets that are covered by a vocabulary.

The use of this module is illustrated in

`scripts.items_coverage <scripts/items_coverage.py>`_




Avoid fetching NMT word embeddings
----------------------------------
2018-12-28

In `scripts.evaluate_embeddings <scripts/evaluate_embeddings.py>`_

the job of evaluating NMT is commented out

because NMT embeddings are no longer available thru the provided url

(this link is broken: https://www.cl.cam.ac.uk/~fh295/TEmbz.tar.gz).




Avoid generator error in Python 3.7
-----------------------------------
2018-12-27

In `web.utils <web/utils.py>`_

in def batched,

replaced

    yield chain([next(batchiter)], batchiter)

by

    try:
        yield chain([next(batchiter)], batchiter)
    except StopIteration:
        return

to void

RuntimeError: generator raised StopIteration

See : Generator raised StopIteration when locateOnScreen

https://stackoverflow.com/questions/51371846/generator-raised-stopiteration-when-locateonscreen/51371879#51371879



Avoid folder creation conflict
------------------------------
2018-12-27

In `web.datasets.utils <web/datasets/utils.py>`_

in def _fetch_helper,

replaced

    os.mkdir(temp_dir)

by

    _makedirs(temp_dir)

to avoid FileExistsError: [Errno 17] File exists

a conflict in folder creation resulting from multiprocessing.




Improve readability
-------------------
2018-12-27

In several places in the code such as

`web.embeddings <web/embeddings.py>`_

print functions

have been added to increase the readibility of the program execution

