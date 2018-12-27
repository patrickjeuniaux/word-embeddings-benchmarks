Major updates
=============

This is a list of changes made to the original project.

Patrick P. J. M. H. Jeuniaux

University of Pisa



4. Avoid generator error in Python 3.7
--------------------------------------
2018-12-27

In web.utils,

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

In web.datasets.utils,

in def _fetch_helper,

replaced

    os.mkdir(temp_dir)

by

    _makedirs(temp_dir)

to avoid FileExistsError: [Errno 17] File exists

a conflict in folder creation resulting from multiprocessing.



2. Add new dataset : SimVerb3500
--------------------------------
2018-12-27

In web.datasets.similarity,

added

def fetch_SimVerb3500

to fetch the SimVerb3500 dataset.

1. Improve readability
----------------------
2018-12-27

In several places,

added

some print functions

to increase the readibility of the program execution

