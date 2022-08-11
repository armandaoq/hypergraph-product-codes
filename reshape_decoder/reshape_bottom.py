from numpy.random import randint
from reshape_decoder.solve_bottom_syndrome_equation import (
    FindSolutionForBottomSyndromeEquation,
)
from gf2.binary_codes import *
from hpc.homological_product import *

"""ErrorCorrection for top syndrome space"""


class ReShapeBottom:
    def __init__(
        self, hp, classical_decoder_delta_a_t=None, classical_decoder_delta_b=None
    ):
        """
        :param hp: homological product instance
        """
        self.hp = hp
        self.classical_decoder_delta_a_t = classical_decoder_delta_a_t
        self.classical_decoder_delta_b = classical_decoder_delta_b
        self.bottom_se_solver = FindSolutionForBottomSyndromeEquation(hp)

        self.ker_a_t = None
        self.ker_b = None

        if classical_decoder_delta_a_t is None:
            self.ker_a_t = find_span(
                hp.gauss_delta_a_t["ker"], [np.zeros(hp.m_a).astype(int)]
            )

            def brute_force_delta_a_t(v):
                choices = [
                    rho
                    for rho in self.ker_a_t
                    if np.sum(((v + rho) % 2))
                    == min(np.sum(((v + k) % 2)) for k in self.ker_a_t)
                ]
                rnd_choice = randint(len(choices))
                return choices[rnd_choice]

            self.classical_decoder_delta_a_t = brute_force_delta_a_t

            # add 0 to the kernel space
        if classical_decoder_delta_b is None:
            self.ker_b = find_span(
                hp.gauss_delta_b["ker"], [np.zeros(hp.n_b).astype(int)]
            )

            def brute_force_delta_b(v):
                choices = [
                    rho
                    for rho in self.ker_b
                    if np.sum(((v + rho) % 2))
                    == min(np.sum(((v + k) % 2)) for k in self.ker_b)
                ]
                rnd_choice = randint(len(choices))
                return choices[rnd_choice]

            self.classical_decoder_delta_b = brute_force_delta_b

    def reshape_decoder(self, syndrome_matrix, valid_solution=None, flag_log=False):
        """Column row decoder:
        1) find a guess for the syndrome_matrix
        2) apply the column-row reduction procedure to
        find the homology class with minimum weight
        Outputs: q_recover. If flag_log outputs the logical part of the recovery too"""
        if valid_solution is None:
            valid_solution = (
                self.bottom_se_solver.find_valid_solution_of_bottom_syndrome_equation(
                    syndrome_matrix
                )
            )

        # find homology class of minimum weight
        left_recovery = []
        right_recovery = []

        for index, row in enumerate(valid_solution["logical_part"].left):
            if any(row):
                rho = self.classical_decoder_delta_b(row)
                left_recovery.append((rho + row) % 2)
            else:
                left_recovery.append(row)
        for index, col in enumerate(valid_solution["logical_part"].right.T):
            if any(col):
                rho = self.classical_decoder_delta_a_t(col)
                right_recovery.append((rho + col) % 2)
            else:
                right_recovery.append(col)

        left_recovery = np.vstack(left_recovery)
        right_recovery = np.vstack(right_recovery)
        # recovery operator
        q_logical_part = Qubits(self.hp, left_recovery, right_recovery.T)
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
                sy, original_error = self.hp.get_random_bottom_syndrome_and_error(
                    prob_of_error=p
                )
                recovery = self.reshape_decoder(sy)
                if not self.hp.assert_success_bottom(original_error, recovery):
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
                ) = self.hp.get_qubit_operator_and_bottom_syndrome_from_vector(o)
                recovery = self.reshape_decoder(sy)
                if not self.hp.assert_success_bottom(original_error, recovery):
                    print("Failure at weight: ", wt, ".\nQubit operator: ")
                    original_error.print_support()
                    return sy, original_error
            print("Weight: ", wt, " successfully decoded.")


if __name__ == "__main__":

    hp_test = Hp(circle_code(4), parity_check_repetition(7))

    print(hp_test)
    from reshape_decoder.solve_bottom_syndrome_equation import (
        FindSolutionForBottomSyndromeEquation,
    )

    my_bottom_class = FindSolutionForBottomSyndromeEquation(hp_test)

    for i in range(1000):
        sy_e, q_e = hp_test.get_random_bottom_syndrome_and_error()
        aa = my_bottom_class.find_valid_solution_of_bottom_syndrome_equation(sy_e)
    reshape_test = ReShapeBottom(hp_test)
    reshape_test.check_decoder_success_on_errors_up_to_wt(5)
