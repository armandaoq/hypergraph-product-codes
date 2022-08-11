from gf2.basics import *
from itertools import combinations


def parity_check_repetition(l):
    """Create delta: l --> l-1 parity check matrix of\
    the repetition code of maximum rank."""
    delta = np.zeros((l, l - 1)).astype(int)
    delta[0][0] = 1
    delta[l - 1][l - 2] = 1
    for i in range(1, l - 1):
        delta[i][i] = 1
        delta[i][i - 1] = 1
    return delta.T


def parity_check_repetition_transpose(l):
    return parity_check_repetition(l).T


def parity_check_double_planar(l):
    padding = np.zeros([l - 1, l], dtype=int)
    left = np.hstack((parity_check_repetition(l), padding))
    right = np.hstack((padding, parity_check_repetition(l)))
    return np.vstack((left, right))


def circle_code(n):
    """Create delta: l --> l parity check matrix of\
        the repetition code of rank l-1.
        delta, delta_t as seeds for homological product give
        the toric code."""
    last_row = np.zeros([1, n]).astype(int)
    last_row[:, 0] = 1
    last_row[:, n - 1] = 1
    d_circle = np.vstack((parity_check_repetition(n), last_row))
    return d_circle


def get_all_vectors_of_weight(w, dim):
    """

    :param w:
    :param dim:
    :return: a generator which iterates on all vectors of weight w and dimension dim
    """

    def get_op(to_flip):
        v = np.zeros([dim], dtype=int)
        v[
            to_flip,
        ] = 1
        return v

    return (get_op(to_flip) for to_flip in combinations(range(dim), w))


def get_span(a, w):
    m, n = a.shape

    def get_op(to_flip):
        v = np.zeros([m], dtype=int)
        v[
            to_flip,
        ] = 1
        return (v @ a) % 2, v

    return (get_op(to_flip) for to_flip in combinations(range(m), w))


def find_distance(a):
    gauss_a = gaussian_reduction(a)
    span_ker = find_span(gauss_a["ker"])
    list_weights = [x.sum() for x in span_ker]
    return min(list_weights)
