from math import factorial


def number_permutations(k, n):
    """
    Calculate the number of permutations of k elements
    chosen in a set of n elements
    """

    # print("permutation of k", k, "n", n)

    if k > n:
        raise ValueError("k cannot be greater than n, but k = {} > n = {}. ".format(k, n))

    return int(factorial(n) / factorial(n - k))
