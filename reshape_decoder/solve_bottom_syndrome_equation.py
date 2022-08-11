from gf2.basics import *
from hpc.homological_product import Qubits


def get_basis_bottom_free_sy_and_qubits(hp):
    """
    :param hp:
    :return: basis of < delta_a_t M delta_b_t | M in n1 x n0 > m as row of a matrix
             indices of the pivot rows, used to find preimage in qubits space
    """
    dim = hp.n_a * hp.m_b
    sy_free = []

    for i in range(hp.m_a):
        for j in range(hp.n_b):
            e = np.zeros([hp.n_a, hp.n_b], dtype=np.int)
            # delta E[i][j] takes column i and put it in col j
            # <--- faster --- E[i][j] = 1
            e[:, j] = hp.delta_a.T[:, i]
            e = (e @ hp.delta_b.T) % 2
            sy_free.append(np.reshape(e, [1, dim]))
    sy_free = np.vstack(sy_free)
    gauss_sy_free = gaussian_reduction(sy_free)

    # return gauss_free_space['im'], gauss_free_space['pivot_row']
    # qubits part
    qubits_free = []  # find basis of preimage of free_syndrome_space
    wt_cols = [sum(c) for c in hp.delta_a]  # so that min weight between left and right
    wt_rows = [sum(r) for r in hp.delta_b.T]
    for p in gauss_sy_free["pivot_row"]:
        left = np.zeros([hp.n_a, hp.n_b], dtype=np.int)
        right = np.zeros([hp.m_a, hp.m_b], dtype=np.int)
        # from p = index of 1 in unit vector back to i, j
        # vector [1, 2, 3, 4, 5, 6] <--> matrix [[1, 2, 3], [4, 5 6]]
        # j = column index = p mod (number of cols)
        # i = row index =  int( p / number of cols)
        # here the syndrome space is n_a x m_b
        # delta E[i][j] takes column i of delta_a and put it in col j
        # E[i][j] takes row j of delta_b and put it on row i
        # E[i][j] belongs is m_a x n_b
        j = p % hp.n_b
        i = int((p - j) / hp.n_b)
        if wt_cols[i] < wt_rows[j]:
            left[:, j] = hp.delta_a.T[:, i]
        else:
            right[i, :] = hp.delta_b.T[j, :]
        qubits_free.append(Qubits(hp, left, right))
    return gauss_sy_free["im"], qubits_free


def get_basis_bottom_logical_sy_and_qubits(hp):
    """
    The left-part of the qubits contributes with k_left = n_a - rank(delta_a) * n_b-rank(delta_b) vectors,
    The right-part of the qubits contributes with k_right = m_b - rank(delta_b) * n_a-rank(delta_a) vectors.
    dim(logical_valid_syndrome_space) = (n_0 - k) * k + (n_1 - k) * k
    :param hp:
    :return: a basis of top logical syndrome space and its preimage
    """
    dim = hp.n_a * hp.m_b
    qubits_logical = []  # find basis of preimage of logical_valid_syndrome_space
    sy_logical = []  # basis of logical valid syndrome space

    # left part
    # splitting is by column in im delta_a.T and complement
    # I need independent rows of delta_b.T
    # supporte on unit vector basis of im delta_a^T^{bot}
    "old is below"
    # slpitting is by row in im delta_b and complement
    # I need independent columns of delta_a,
    # supported on unit vector basis of (im delta_b^T)^{bot}
    pivot_rows_delta_b = hp.gauss_delta_b_t["pivot_row"]
    left_out = hp.gauss_delta_a["complete_to_basis"]
    id_0_b = np.identity(hp.n_b, dtype=np.int)  # needed to find preimage
    for f in left_out:
        for j in pivot_rows_delta_b:
            left = np.outer(f, id_0_b[j])
            syndrome = (left @ hp.delta_b.T) % 2
            sy_logical.append(np.reshape(syndrome, [1, dim]))
            qubits_logical.append(Qubits(hp, left, None))
    # right part
    # splitting is by column in im delta_a
    # I need independent rows of delta_b,
    # supported on unit vector basis of (Im delta_a)^{bot}
    pivot_cols_delta_a_t = hp.gauss_delta_a["pivot_row"]
    right_out = hp.gauss_delta_b_t["complete_to_basis"]
    id_1_a = np.identity(hp.m_a, dtype=np.int)
    for f in right_out:
        for i in pivot_cols_delta_a_t:
            right = np.outer(id_1_a[i], f)
            syndrome = (hp.delta_a.T @ right) % 2
            sy_logical.append(np.reshape(syndrome, [1, dim]))
            qubits_logical.append(Qubits(hp, None, right))
    sy_logical = np.vstack(sy_logical)
    assert sy_logical.shape[0] == rank(
        sy_logical
    ), "Need a basis for the logical " "syndrome space, have found " + str(
        rank(sy_logical)
    ) + "independent vectors out of " + str(
        sy_logical.shape[0]
    )
    return sy_logical, qubits_logical


class FindSolutionForBottomSyndromeEquation:
    def __init__(self, hp):

        sy_free, qubits_free = get_basis_bottom_free_sy_and_qubits(hp)
        sy_logical, qubits_logical = get_basis_bottom_logical_sy_and_qubits(hp)
        sy_valid = np.vstack([sy_free, sy_logical])
        qubits_valid = qubits_free + qubits_logical
        gauss_sy_valid = gaussian_reduction(sy_valid)  # need complete_to_basis
        # from free plus logical syndrome basis B to canonical C
        # change_of_basis * [v]_B = [v]_C
        change_of_basis = np.vstack([sy_valid, gauss_sy_valid["complete_to_basis"]]).T

        self.hp = hp
        self.dim_free_sy_space = sy_free.shape[0]
        # from canonical to free plus logical basis
        # change_of_basis * [v]_C = [v]_B
        self.change_of_basis_syndrome = inverse(change_of_basis)
        self.qubits_valid = qubits_valid

    def find_valid_solution_of_bottom_syndrome_equation(self, syndrome_matrix):
        """
        Finds a qubit operator whose syndrome is equal to syndrome_matrix,
        considering a top syndrome equation
        :param syndrome_matrix:
        :return: a dictionary
        """
        syndrome_vector = np.reshape(
            syndrome_matrix, [1, self.hp.dim_syndrome_space_bottom]
        )
        # rewrite sydnrome in free plus logical basis
        syndrome_free_plus_logical_basis = (
            self.change_of_basis_syndrome @ syndrome_vector.T % 2
        )
        # error is written in std basis
        free_part_error = [
            self.qubits_valid[i]
            for i in range(self.dim_free_sy_space)
            if syndrome_free_plus_logical_basis[i]
        ]
        logical_part_error = [
            self.qubits_valid[i]
            for i in range(self.dim_free_sy_space, self.hp.dim_syndrome_space_bottom)
            if syndrome_free_plus_logical_basis[i]
        ]

        q_free_part = Qubits(self.hp)
        for q in free_part_error:
            q_free_part += q

        q_logical_part = Qubits(self.hp)
        for q in logical_part_error:
            q_logical_part += q

        q_solution = q_free_part + q_logical_part
        assert q_logical_part.left.any(axis=1).sum() <= (
            self.hp.n_a - self.hp.gauss_delta_a["rank"]
        ), print("logical left part supported on too many rows")
        assert (
            q_logical_part.right.any(axis=0).sum()
            <= self.hp.m_b - self.hp.gauss_delta_b["rank"]
        ), print("logical right part supported on too many columns")

        assert np.array_equal(syndrome_matrix, self.hp.bottom_syndrome(q_solution))
        return {
            "free_part": q_free_part,
            "logical_part": q_logical_part,
            "solution": q_solution,
        }
