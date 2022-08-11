from numpy.random import randint
from reshape_decoder.solve_top_syndrome_equation import (
    FindSolutionForTopSyndromeEquation,
)
from gf2.binary_codes import *
from hpc.homological_product import *

"""ErrorCorrection for top syndrome space"""


class ReShapeTop:
    def __init__(
        self, hp, classical_decoder_delta_a=None, classical_decoder_delta_b_t=None
    ):
        """
        :param hp: homological product instance
        """
        self.hp = hp
        self.classical_decoder_delta_a = classical_decoder_delta_a
        self.classical_decoder_delta_b_t = classical_decoder_delta_b_t
        self.top_se_solver = FindSolutionForTopSyndromeEquation(hp)

        self.ker_a = None
        self.ker_b_t = None

        if classical_decoder_delta_a is None:
            self.ker_a = find_span(
                hp.gauss_delta_a["ker"], [np.zeros(hp.n_a).astype(int)]
            )

            def brute_force_delta_a(v):
                choices = [
                    rho
                    for rho in self.ker_a
                    if np.sum(((v + rho) % 2))
                    == min(np.sum(((v + k) % 2)) for k in self.ker_a)
                ]
                rnd_choice = randint(len(choices))
                return choices[rnd_choice]

            self.classical_decoder_delta_a = brute_force_delta_a

            # add 0 to the kernel space
        if classical_decoder_delta_b_t is None:
            self.ker_b_t = find_span(
                hp.gauss_delta_b_t["ker"], [np.zeros(hp.m_b).astype(int)]
            )

            def brute_force_delta_b_t(v):
                choices = [
                    rho
                    for rho in self.ker_b_t
                    if np.sum(((v + rho) % 2))
                    == min(np.sum(((v + k) % 2)) for k in self.ker_b_t)
                ]
                rnd_choice = randint(len(choices))
                return choices[rnd_choice]

            self.classical_decoder_delta_b_t = brute_force_delta_b_t

    def reshape_decoder(self, syndrome_matrix, valid_solution=None, flag_log=False):
        """Column row decoder:
        1) find a guess for the syndrome_matrix
        2) apply the column-row reduction procedure to
        find the homology class with minimum weight
        Outputs: q_recover. If flag_log outputs the logical part of the recovery too"""
        if valid_solution is None:
            valid_solution = (
                self.top_se_solver.find_valid_solution_of_top_syndrome_equation(
                    syndrome_matrix
                )
            )

        # find homology class of minimum weight
        left_recovery = []
        right_recovery = []

        for index, col in enumerate(valid_solution["logical_part"].left.T):
            if any(col):
                rho = self.classical_decoder_delta_a(col)
                left_recovery.append((rho + col) % 2)
            else:
                left_recovery.append(col)
        for index, row in enumerate(valid_solution["logical_part"].right):
            if any(row):
                rho = self.classical_decoder_delta_b_t(row)
                right_recovery.append((rho + row) % 2)
            else:
                right_recovery.append(row)

        left_recovery = np.vstack(left_recovery)
        right_recovery = np.vstack(right_recovery)
        # recovery operator
        q_logical_part = Qubits(self.hp, left_recovery.T, right_recovery)
        q_recovery = valid_solution["free_part"] + q_logical_part
        if not flag_log:
            return q_recovery
        else:
            return q_recovery, q_logical_part

    def run_experiment(self, list_tuple_p_n):
        """

        :param list_tuple_p_n:
        :return:
        """

        list_tuple_p_n_failures = []
        for (p, n) in list_tuple_p_n:
            failures = 0
            for i in range(n):
                if i % 100 == 0:
                    print(
                        p,
                        "prob of error, ",
                        i,
                        " number of samples -- number of failures: ",
                        failures,
                    )
                sy, original_error = self.hp.get_random_top_syndrome_and_error(
                    prob_of_error=p
                )
                recovery = self.reshape_decoder(sy)
                if not self.hp.assert_success_top(original_error, recovery):
                    failures += 1
            print("*********** ", p, failures, n)
            list_tuple_p_n_failures.append((p, failures, n))
        return list_tuple_p_n_failures

    def check_decoder_success_on_errors_up_to_wt(self, max_wt, min_wt=1):
        """
        :param max_wt:
        :param min_wt:
        :return:
        """
        for wt in range(min_wt, max_wt + 1):
            print("---")
            operators = get_all_vectors_of_weight(wt, self.hp.n_qubit)
            for o in operators:
                (
                    original_error,
                    sy,
                ) = self.hp.get_qubit_operator_and_top_syndrome_from_vector(o)
                recovery = self.reshape_decoder(sy)
                if not self.hp.assert_success_top(original_error, recovery):
                    print("Failure at weight: ", wt, ".\nQubit operator: ")
                    original_error.print_support()
                    return sy, original_error
            print("Weight: ", wt, " successfully decoded.")


if __name__ == "__main__":
    from codes.input_for_testing import hamming_deg

    hp = Hp(hamming_deg, hamming_deg)
    reshape = ReShapeTop(hp)
    reshape.check_decoder_success_on_errors_up_to_wt(2)
    sy_, original_error_ = reshape.hp.get_random_top_syndrome_and_error()
    recovery_ = reshape.reshape_decoder(sy_)
