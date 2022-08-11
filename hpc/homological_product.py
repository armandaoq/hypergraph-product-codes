import numpy as np
from numpy.random import random
from gf2.basics import gaussian_reduction
from hpc.qubit_space import Qubits

"""Homological Product Code with C_a0 x C_b1 --> C_a0 x C_b0 + C_a1 x C_b1 --> C_a1 x C_b0
   logical bottom-operators are vectors in ker H_{top} not in image H_{bottom}^T
   logical top-operators are vectors in ker H_{bottom} not in  image H_{top}^T"""


class Hp:
    def __init__(self, delta_a, delta_b, params=True):
        self.params = params
        self.delta_a = delta_a
        self.delta_b = delta_b
        self.m_a, self.n_a = delta_a.shape
        self.m_b, self.n_b = delta_b.shape
        self.n_left = self.n_a * self.n_b
        self.n_right = self.m_a * self.m_b
        self.n_qubit = self.n_left + self.n_right
        self.dim_syndrome_space_top = None
        self.dim_syndrome_space_bottom = None
        self.k_q = None
        self.k_left = None
        self.k_right = None
        self.rk_sy_top = None
        self.rk_sy_bot = None
        self.gauss_delta_a = None
        self.gauss_delta_a_t = None
        self.gauss_delta_b = None
        self.gauss_delta_b_t = None
        self.matrix_logicals_top = None
        self.matrix_logicals_bottom = None
        if self.params:
            self.set_params()

    def __repr__(self):
        # out = 'n_0: ' + str(self.n0) + ', n_1: ' + str(self.n1) + ',\ndelta:\n' + str(self.delta)
        out = "n_a: " + str(self.n_a) + ", m_a: " + str(self.m_a)
        out += " n_b: " + str(self.n_b) + ", m_b: " + str(self.m_b)
        out += "\nlength: " + str(self.n_qubit) + ", dimension: " + str(self.k_q) + "\n"
        out += "k left: " + str(self.k_left) + ", k right: " + str(self.k_right) + "\n"
        out += "dimension valid syndrome space top: " + str(self.rk_sy_top)
        out += ", dimension valid syndrome space bottom: " + str(self.rk_sy_bot)
        out += (
            ",\ndimension syndrome space top: "
            + str(self.m_a * self.n_b)
            + ", bottom: "
            + str(self.n_a * self.m_b)
        )
        return out

    def set_params(self):
        """N = length of the code, K  = dimension"""
        self.gauss_delta_a = gaussian_reduction(self.delta_a)
        self.gauss_delta_a_t = gaussian_reduction(self.delta_a.T)
        self.gauss_delta_b = gaussian_reduction(self.delta_b)
        self.gauss_delta_b_t = gaussian_reduction(self.delta_b.T)
        k_a, k_b = (self.n_a - self.gauss_delta_a["rank"]), (
            self.n_b - self.gauss_delta_b["rank"]
        )
        k_a_t, k_b_t = (self.m_a - self.gauss_delta_a["rank"]), (
            self.m_b - self.gauss_delta_b["rank"]
        )
        self.k_left = k_a * k_b
        self.k_right = k_a_t * k_b_t
        self.k_q = (k_a * k_b) + (k_a_t * k_b_t)
        assert self.k_q > 0, "Dimension of the code: " + str(self.k_q)
        self.dim_syndrome_space_top = self.m_a * self.n_b
        self.dim_syndrome_space_bottom = self.n_a * self.m_b
        self.rk_sy_top = self.m_a * self.n_b - k_a_t * k_b  # check!
        self.rk_sy_bot = self.n_a * self.m_b - k_a * k_b_t  # check!
        self.matrix_logicals_top = self.get_matrix_logicals_top()
        self.matrix_logicals_bottom = self.get_matrix_logicals_bottom()

    def bottom_stabiliser(self, a, beta):
        """

        :param a:
        :param beta:
        :return: bottom stabiliser qubit operator
        """
        e = np.zeros([self.n_a, self.m_b]).astype(int)
        e[a][beta] = 1
        g_left = (e @ self.delta_b) % 2
        g_right = (self.delta_a @ e) % 2
        return Qubits(self, g_left, g_right)

    def top_stabilizer(self, beta, a):
        """
        :param beta:
        :param a:
        :return: top stabiliser qubit operator
        """
        e = np.zeros([self.m_a, self.n_b]).astype(int)
        e[beta][a] = 1
        g_left = (self.delta_a.T @ e) % 2
        g_right = (e @ self.delta_b.T) % 2
        return Qubits(self, g_left, g_right)

    def top_syndrome(self, q):
        """
        Measure top stabilisers
        :param q:
        :return:
        """
        s = (self.delta_a @ q.left + q.right @ self.delta_b) % 2
        return s

    def bottom_syndrome(self, q):
        """
        Measure bottom stabilisers
        :param q:
        :return:
        """
        """measure top operators"""
        s = (q.left @ self.delta_b.T + self.delta_a.T @ q.right) % 2
        return s

    def get_qubit_operator_and_top_syndrome_from_vector(self, e):
        """

        :param e: vector form qubit error operator
        :return: qubit operator and its top syndrome matrix
        """
        left = np.reshape(e[: self.n_left], [self.n_a, self.n_b])
        right = np.reshape(e[self.n_left :], [self.m_a, self.m_b])
        q_e = Qubits(self, left, right)
        return q_e, self.top_syndrome(q_e)

    def get_qubit_operator_and_bottom_syndrome_from_vector(self, e):
        """

        :param e: vector form qubit error operator
        :return: qubit operator and its top syndrome matrix
        """
        left = np.reshape(e[: self.n_left], [self.n_a, self.n_b])
        right = np.reshape(e[self.n_left :], [self.m_a, self.m_b])
        q_e = Qubits(self, left, right)
        return q_e, self.bottom_syndrome(q_e)

    def get_matrix_logicals_bottom(self):
        """
        A basis of the logical bottom operators is given by
        left = {ker delta_a x (Im delta_b_t)^{bot}, 0}
        right = {0, (Im delta_a)^{bot} x ker delta_b_t}
        :return: matrix whose rows are logical operators of the described form
        """

        left_out = self.gauss_delta_b[
            "complete_to_basis"
        ]  # complete to basis looks at rows
        right_out = self.gauss_delta_a_t["complete_to_basis"]
        log = np.empty((0, self.n_left + self.n_right)).astype(int)
        for f in left_out:
            for k in self.gauss_delta_a["ker"]:
                left = np.outer(k, f) % 2
                logical_op = np.hstack(
                    [
                        np.reshape(left, [1, self.n_left]),
                        np.zeros([1, self.n_right]).astype(int),
                    ]
                )
                log = np.vstack((log, logical_op))
        for f in right_out:
            for k in self.gauss_delta_b_t["ker"]:
                right = np.outer(f, k) % 2
                logical_op = np.hstack(
                    [
                        np.zeros([1, self.n_left]).astype(int),
                        np.reshape(right, [1, self.n_right]),
                    ]
                )
                log = np.vstack((log, logical_op))
        return log

    def get_matrix_logicals_top(self):
        """
        A basis of the logical top operators given by
        left = {(Im delta_a_t)^{bot} x ker delta_b, 0}
        right = {0, ker delta_a_t x (Im delta_b)^{bot}}
        :return: matrix whose rows are logical operators of the described form
        """

        left_out = self.gauss_delta_a["complete_to_basis"]
        right_out = self.gauss_delta_b_t["complete_to_basis"]
        log = np.empty((0, self.n_left + self.n_right)).astype(int)
        for k in self.gauss_delta_b["ker"]:
            for f in left_out:
                left = np.outer(f, k) % 2
                logical_op = np.hstack(
                    (
                        np.reshape(left, [1, self.n_left]),
                        np.zeros([1, self.n_right]).astype(int),
                    )
                )
                log = np.vstack((log, logical_op))
        for k in self.gauss_delta_a_t["ker"]:
            for f in right_out:
                right = np.outer(k, f) % 2
                logical_op = np.hstack(
                    (
                        np.zeros([1, self.n_left]).astype(int),
                        np.reshape(right, [1, self.n_right]),
                    )
                )
                log = np.vstack((log, logical_op))
        return log

    def basis_of_logicals_top(self):
        """ "returns a list of qubits operators that are a basis of logical top space"""
        if not self.params:
            self.set_params()
        list_of_logical = []
        for logical_vec in self.matrix_logicals_top:
            qubit_op = Qubits(self)
            list_of_logical.append(qubit_op.from_vector(logical_vec))
        return list_of_logical

    def basis_of_logicals_bottom(self):
        """ "returns a list of qubits operators that are a basis of logical bottom space"""
        if not self.params:
            self.set_params()
        list_of_logical = []
        for logical_vec in self.matrix_logicals_bottom:
            qubit_op = Qubits(self)
            list_of_logical.append(qubit_op.from_vector(logical_vec))
        return list_of_logical

    def check_homology_top(self, o):
        """input: op top-operator, check commutation with all logicals top
        In fact, if we measure the top syndrome, we should check commutation with all logical top
        output: relations[i] = 1  iff operators op anti-commute with i-th logical top op of basis"""
        if not self.params:
            self.set_params()
        o_vector = np.hstack(
            (
                np.reshape(o.left, [1, self.n_left]),
                np.reshape(o.right, [1, self.n_right]),
            )
        )
        relations = (self.matrix_logicals_top @ o_vector.T) % 2
        return relations.T

    def check_homology_bottom(self, o):
        if not self.params:
            self.set_params()
        o_vector = np.hstack(
            (
                np.reshape(o.left, [1, self.n_left]),
                np.reshape(o.right, [1, self.n_right]),
            )
        )
        relations = (self.matrix_logicals_bottom @ o_vector.T) % 2
        return relations.T

    def assert_success_top(self, q_recovery, q_e):
        """

        :param q_recovery:
        :param q_e:
        :return: True iff q_recovery and q_e are in the same bottom-homolgy class
        meaning that they respect the same commutation relations with a basis
        of logical top operators
        """
        relations = self.check_homology_top(q_recovery + q_e)
        return not relations.any()  # vec.any() False iff all zero

    def assert_success_bottom(self, q_recovery, q_e):
        """

        :param q_recovery:
        :param q_e:
        :return: True iff q_recovery and q_e are in the same top-homolgy class
        meaning that they respect the same commutation relations with a basis
        of logical bottom operators
        """

        relations = self.check_homology_bottom(q_recovery + q_e)
        return not relations.any()  # vec.any() False iff all zero

    def get_zero_syndrome_top(self):
        return np.zeros([self.m_a, self.n_b], dtype=np.int)

    def get_zero_syndrome_bottom(self):
        return np.zeros([self.n_a, self.m_b], dtype=np.int)

    def get_random_error_vector(self, prob_of_error):
        error = random([self.n_qubit])
        error = np.array(
            [1 if error[i] < prob_of_error else 0 for i in range(self.n_qubit)],
            dtype=np.int,
        )
        return error

    def get_random_top_syndrome_and_error(self, prob_of_error=0.025):
        e = self.get_random_error_vector(prob_of_error=prob_of_error)
        q_e = Qubits(self)
        q_e.from_vector(e)
        sy = self.top_syndrome(q_e)
        return sy, q_e

    def get_random_bottom_syndrome_and_error(self, prob_of_error=0.025):
        e = self.get_random_error_vector(prob_of_error=prob_of_error)
        q_e = Qubits(self)
        q_e.from_vector(e)
        sy = self.bottom_syndrome(q_e)
        return sy, q_e

    def get_qubit_operator_from_vector(self, vec):
        q_e = Qubits(self)
        return q_e.from_vector(vec)


if __name__ == "__main__":
    from codes.input_for_testing import hamming_deg
    from gf2.binary_codes import circle_code, parity_check_repetition

    print("\nPlanar code")
    hp_planar = Hp(parity_check_repetition(3), parity_check_repetition(3))
    print(hp_planar)
    print("\nToric code")
    hp_toric = Hp(circle_code(3), circle_code(3))
    print(hp_toric)

    hp_hamming_deg = Hp(hamming_deg, hamming_deg)
    print("\nHamming degenerate:")
    print(hp_hamming_deg)
    print('\nIt is the hypergraph product of the following matrix with itself\n', hamming_deg)
    print('Here a strongly reduced basis for the kernel of the Hamming degenerate matrix that yields a canonical \n'
          'form for the logical operators')
    print(hp_hamming_deg.gauss_delta_a['ker'])
    for i, q in enumerate(hp_hamming_deg.basis_of_logicals_top()):
        print(i, "anticommuting pairs of logicals -----------------------")
        print(q)
        print('*')
        print(hp_hamming_deg.basis_of_logicals_bottom()[i])
