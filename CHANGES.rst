Main updates
============

This is a list of changes made to the original project.

Patrick Jeuniaux

University of Pisa



6. Count the number of items
----------------------------
2018-12-31 --- 2019-01-02

Add `web.datasets.items <web/datasets/items.py>`_

to count the number of items in the datasets.

The use of this module is illustrated in

`scripts.count_items <scripts/count_items.py>`_



5. Avoid fetching NMT word embeddings
-------------------------------------
2018-12-28

In `scripts.evaluate_embeddings <scripts/evaluate_embeddings.py>`_

the job of evaluating NMT is commented out

because NMT embeddings are no longer available thru the provided url

(this link is broken: https://www.cl.cam.ac.uk/~fh295/TEmbz.tar.gz).

4. Avoid generator error in Python 3.7
--------------------------------------
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



3. Avoid folder creation conflict
---------------------------------
2018-12-27

In `web.datasets.utils <web/datasets/utils.py>`_

in def _fetch_helper,

replaced

    os.mkdir(temp_dir)

by

    _makedirs(temp_dir)

to avoid FileExistsError: [Errno 17] File exists

a conflict in folder creation resulting from multiprocessing.



2. Add fetch functions for new datasets
---------------------------------------

- 2018-12-27 : SimVerb3500
- 2019-01-01 : SAT
- 2019-01-01 : BATS
- 2019-01-02 : ESL
- 2019-01-03 : TOEFL


In `web.datasets.similarity <web/datasets/similarity.py>`_

added

    def fetch_SimVerb3500


In `web.datasets.analogy <web/datasets/analogy.py>`_

added

    def fetch_SAT
    def fetch_BATS


In `web.datasets.synonymy <web/datasets/synonymy.py>`_

added

    def fetch_ESL
    def fetch_TOEFL


Note:

except for SimVerb3500, the datasets are not downloaded

from the Internet but from the local machine.

TODO:

find official online versions of the datasets


1. Improve readability
----------------------
2018-12-27

In several places in the code such as

`web.embeddings <web/embeddings.py>`_

added

print functions

to increase the readibility of the program execution

