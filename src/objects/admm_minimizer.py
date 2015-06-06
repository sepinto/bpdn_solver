import numpy as np
from numpy.linalg import norm

from src.objects.minimizer import Minimizer
from src.functions.matrices import left_multiply_Z, Z_diag_ZT, create_Z


class ADMMMinimizer(Minimizer):
    def __init__(self, coding_params, weights, admm_lambda, y, phi_indices):
        Minimizer.__init__(self, coding_params, weights, np.zeros(coding_params.full_block_length), lambd=1.0 / (2.0 * admm_lambda), y=y,
                           phi_indices=phi_indices)
        self.admm_lambda = admm_lambda
        self.x = np.zeros(self.n)
        self.relaxed_term = np.copy(self.x)
        self.z = np.zeros(self.n)
        self.prev_z = np.zeros(self.n)
        self.u = np.zeros(self.n)
        self.optimal = False
        self.relative_epsilon = 10 ** -3.5
        self.iterations = 0
        self.max_iters = 1000

        # Precompute some values
        self.weights_inv_sqr = np.power(self.weights, -2.0)
        self.rank_m_inverse = self.get('rank_m_inverse')
        self.x_update_constant = self.left_multiply_update_inverse(
            left_multiply_Z(self.y, self.phi_indices, self.n, transpose=True))
        self.soft_threshold_constant = self.admm_lambda / self.coding_params.rho

    def calc_update_matrix(self):
        Z = create_Z(self.phi_indices, self.n)
        return np.dot(Z.transpose(), Z) + self.coding_params.rho * np.diag(np.power(self.weights, 2.0))

    def calc_update_matrix_inv(self):
        return np.asarray(np.linalg.inv(self.get('update_matrix')))

    def calc_rank_m_matrix(self):
        rank_m_matrix = Z_diag_ZT(self.weights_inv_sqr, self.phi_indices)
        np.fill_diagonal(rank_m_matrix, np.diag(rank_m_matrix) + self.coding_params.rho)
        return rank_m_matrix

    def calc_rank_m_inverse(self):
        return np.asarray(np.linalg.inv(self.get('rank_m_matrix')))

    def solve(self):
        while not self.optimal:
            self.iterate()
        return self.x, self.objective(self.x)

    def iterate(self):
        self.x_update()
        self.z_update()
        self.u_update()
        self.opt_update()
        self.iterations += 1

    def x_update(self):
        iter_term = self.coding_params.rho * self.weights * (self.z - self.u)
        self.x = self.x_update_constant + self.left_multiply_update_inverse(iter_term)

    def z_update(self):
        self.prev_z = np.copy(self.z)
        self.relaxed_term = self.coding_params.alpha_admm * self.weights * self.x + (1 - self.coding_params.alpha_admm) * self.prev_z
        # self.x_relaxed = self.coding_params.alpha_admm * self.x + (1.0 - self.coding_params.alpha_admm) * self.prev_z
        self.z = self.soft_threshold(self.relaxed_term + self.u)

    def u_update(self):
        self.u = self.u + self.relaxed_term - self.z

    def primal_residual(self):
        return self.weights * self.x - self.z

    def dual_residual(self):
        return self.coding_params.rho * self.weights * (self.z - self.prev_z)

    def primal_epsilon(self):
        return np.sqrt(self.n) * self.coding_params.epsilon + \
               self.relative_epsilon * np.max((norm(self.weights * self.x), norm(self.z)))

    def dual_epsilon(self):
        return np.sqrt(self.n) * self.coding_params.epsilon + \
               self.relative_epsilon * norm(self.weights * self.u) * self.coding_params.rho

    def opt_update(self):
        self.optimal = norm(self.primal_residual(), 2) <= self.primal_epsilon() and norm(self.dual_residual(), 2) <= self.dual_epsilon()
        self.optimal = self.optimal or self.iterations > self.max_iters

    def soft_threshold(self, nu):
        return np.maximum(0, nu - self.soft_threshold_constant) - np.maximum(0, -1 * nu - self.soft_threshold_constant)

    def left_multiply_update_inverse(self, x):
        weights_inv_x = self.weights_inv_sqr * x
        result = np.copy(weights_inv_x)
        result = left_multiply_Z(result, self.phi_indices, self.n, transpose=False)
        result = np.dot(self.rank_m_inverse, result)
        result = left_multiply_Z(result, self.phi_indices, self.n, transpose=True)
        result = self.weights_inv_sqr * result
        return (weights_inv_x - result) / self.coding_params.rho
