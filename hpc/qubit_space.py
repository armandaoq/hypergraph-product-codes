import numpy as np


class Qubits:
    """ "class that represents qubits operator on C_0xC_0 + C_1xC_1"""

    def __init__(self, hp, left=None, right=None):
        self.hp = hp
        if left is None:
            self.left = np.zeros([hp.n_a, hp.n_b], dtype=np.int)
        else:
            self.left = np.copy(left)
        if right is None:
            self.right = np.zeros([hp.m_a, hp.m_b], dtype=np.int)
        else:
            self.right = np.copy(right)

    def __repr__(self):
        return "\n(" + str(self.left) + ",\n" + str(self.right) + ")"

    def __add__(self, other):
        left = (self.left + other.left) % 2
        right = (self.right + other.right) % 2
        return Qubits(self.hp, left, right)

    def __eq__(self, other):
        return np.array_equal(self.left, other.left) and np.array_equal(
            self.right, other.right
        )

    def integer_sum(self):
        return self.left + self.right

    def print_support(self):
        left = sorted(
            {
                (i, j)
                for i in range(self.hp.n_a)
                for j in range(self.hp.n_b)
                if self.left[i, j]
            }
        )
        right = sorted(
            {
                (i, j)
                for i in range(self.hp.m_a)
                for j in range(self.hp.m_b)
                if self.right[i, j]
            }
        )
        print(left, "<--- Left part -- Right part --->", right)

    def is_zero(self):
        return not (self.left.any() or self.right.any())

    def weight(self):
        return np.count_nonzero(self.left) + np.count_nonzero(self.right)

    def to_vector(self):
        vec_left = np.reshape(self.left, [self.hp.n_left])
        vec_right = np.reshape(self.right, [self.hp.n_right])
        return np.hstack([vec_left, vec_right])

    def from_vector(self, vec):
        vec_left = vec[: self.hp.n_left]
        vec_right = vec[self.hp.n_left :]
        self.left = np.copy(np.reshape(vec_left, [self.hp.n_a, self.hp.n_b]))
        self.right = np.copy(np.reshape(vec_right, [self.hp.m_a, self.hp.m_b]))
        return self

    def from_support_set(self, support_set):
        """
        set is a diction
        :param support_set: set is a dictionary with keys 'left' and 'right'
        set['left'] is a set of tuple (i, j) which constitutes the support
        of the left part of the qubit operator; similar for set['right'].
        :return: qubit operator with support set['left'] on the left part and set['right']
        on the right part
        """
        for (i, j) in support_set["left"]:
            self.left[i, j] = 1
        for (i, j) in support_set["right"]:
            self.right[i, j] = 1
        return self
