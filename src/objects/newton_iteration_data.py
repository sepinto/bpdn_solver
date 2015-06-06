import numpy as np
from src.functions.matrices import left_multiply_Z, Z_diag_ZT

class NewtonIterationData:
    def __init__(self, z, bm_minimizer):
        self.bm_minimizer = bm_minimizer
        self.n = self.bm_minimizer.n # because it's accessed so much. Shorter to write this way.
        self.m = self.bm_minimizer.m
        self.z = np.copy(z)

        # Ensure we only the absolutely necessary calculations
        self.vals = {}

    def get(self, key):
        if key in self.vals:
            return self.vals[key]
        else:
            val = getattr(self, 'calc_' + key)()
            self.vals[key] = np.copy(val)
            return val

    def feasible(self):
        return self.get('feasible')

    def calc_feasible(self):
        return self.bm_minimizer.feasible_z(self.z)

    def calc_Zx(self):
        return left_multiply_Z(self.z[0:self.n], self.bm_minimizer.phi_indices, self.n) if self.bm_minimizer.quadratic else None

    ##### The Objective & It's Derivatives #####
    def calc_objective(self):
        return self.bm_minimizer.z_objective(self.z, self.get('Zx'))

    def calc_grad_one_norm_obj(self):
        return np.concatenate((np.zeros(self.n), np.ones(self.n)))

    def calc_grad_two_norm_obj(self):
        gradx = 2.0 * left_multiply_Z(self.get('Zx'), self.bm_minimizer.phi_indices, self.n, transpose=True) - 2.0 * self.bm_minimizer.ZTy
        return np.concatenate((gradx, np.zeros(self.n)))

    def calc_grad_objective(self):
        return self.calc_grad_one_norm_obj() + (self.bm_minimizer.lambd * self.get('grad_two_norm_obj') if self.bm_minimizer.quadratic else 0)

    def calc_hess_two_norm_obj(self):
        return 2.0 * np.asarray(np.bmat(([np.dot(self.bm_minimizer.get('Z').transpose(), self.bm_minimizer.get('Z')),
                                          np.zeros((self.n, self.n))], [np.zeros((self.n, 2 * self.n))])))

    def calc_hess_objective(self):
        return self.bm_minimizer.lambd * self.get('hess_two_norm_obj') if self.bm_minimizer.quadratic else np.zeros((2 * self.n, 2 * self.n))

    ##### The Barrier Fcn & It's Derivatives #####
    def calc_lin_ineq_constraint(self):
        return self.bm_minimizer.linear_inequality_constraint(self.z)

    def calc_barrier_fcn(self):
        return np.sum(-1.0 * np.log(-1.0 * self.get('lin_ineq_constraint')))

    def calc_grad_barrier_fcn(self):
        first_half = 1.0 / self.get('lin_ineq_constraint')[0:self.n]
        second_half = 1.0 / self.get('lin_ineq_constraint')[self.n:]
        return np.concatenate((self.bm_minimizer.weights * (second_half - first_half), first_half + second_half))
        # return - 1.0 * np.asarray(np.dot(self.bm_minimizer.A.transpose(), 1.0 / self.get('lin_ineq_constraint')))[0] # TODO: This can be way faster

    def calc_hess_barrier_fcn(self):
        first_half_inv_squared = 1.0 / (np.power(self.get('lin_ineq_constraint')[0:self.n], 2.0))
        second_half_inv_squared = 1.0 / (np.power(self.get('lin_ineq_constraint')[self.n:], 2.0))
        d1 = np.power(self.bm_minimizer.weights, 2.0) * (first_half_inv_squared + second_half_inv_squared)
        d2 = self.bm_minimizer.weights * (second_half_inv_squared - first_half_inv_squared)
        d3 = first_half_inv_squared + second_half_inv_squared
        return np.asarray(np.bmat(([np.diag(d1), np.diag(d2)],
                                   [np.diag(d2), np.diag(d3)])))

    def calc_d1(self):
        first_half_sqr = np.power(self.get('lin_ineq_constraint')[0:self.n], 2.0)
        second_half_sqr = np.power(self.get('lin_ineq_constraint')[self.n:], 2.0)
        return (first_half_sqr + second_half_sqr) / (4.0 * np.power(self.bm_minimizer.weights, 2.0))

    def calc_d2(self):
        first_half_sqr = np.power(self.get('lin_ineq_constraint')[0:self.n], 2.0)
        second_half_sqr = np.power(self.get('lin_ineq_constraint')[self.n:], 2.0)
        return (second_half_sqr - first_half_sqr) / (4.0 * self.bm_minimizer.weights)

    def calc_d3(self):
        first_half_sqr = np.power(self.get('lin_ineq_constraint')[0:self.n], 2.0)
        second_half_sqr = np.power(self.get('lin_ineq_constraint')[self.n:], 2.0)
        return (first_half_sqr + second_half_sqr) / 4.0

    def calc_hess_barrier_fcn_inv(self):
        return np.asarray(np.bmat(([np.diag(self.get('d1')), np.diag(self.get('d2'))],
                                   [np.diag(self.get('d2')), np.diag(self.get('d3'))])))

    def left_multiply_hess_barrier_fcn_inv(self, input):
        first_half = (self.get('d1') * input[0:self.n] + self.get('d2') * input[self.n:])
        second_half = (self.get('d2') * input[0:self.n] + self.get('d3') * input[self.n:])
        return np.concatenate((first_half, second_half))

    ##### The Newton Obj & It's Derivatives #####
    def calc_newton_obj(self):
        return self.bm_minimizer.t * self.get('objective') + self.get('barrier_fcn')

    def calc_grad_newton_obj(self):
        return self.bm_minimizer.t * self.get('grad_objective') + self.get('grad_barrier_fcn')

    def calc_hess_newton_obj(self):
        return self.bm_minimizer.t * self.get('hess_objective') + self.get('hess_barrier_fcn')

    def calc_hess_newton_obj_inv(self):
        return np.linalg.inv(self.get('hess_newton_obj'))

    def calc_rank_m_matrix_inverse(self): # O(mn^2)
        to_be_inverted = Z_diag_ZT(self.get('d1'), self.bm_minimizer.phi_indices)
        np.fill_diagonal(to_be_inverted, np.diag(to_be_inverted) + 1.0 / (2.0 * self.bm_minimizer.lambd * self.bm_minimizer.t))
        inv = np.asarray(np.linalg.inv(to_be_inverted))
        return inv

    def matrix_inversion_lemma(self):
        Az_inv = self.get('hess_barrier_fcn_inv')
        U = np.asarray(np.bmat(([self.bm_minimizer.get('Z').transpose()], [np.zeros((self.n, self.m))])))
        result = np.dot(Az_inv, np.dot(U, np.dot(self.get('rank_m_matrix_inverse'), np.dot(U.transpose(), Az_inv))))
        return np.asarray(Az_inv - result)

    def left_multiply_hess_newton_obj_inv(self, input):
        if self.bm_minimizer.quadratic:
            Az_inv_input = self.left_multiply_hess_barrier_fcn_inv(input)
            result = np.copy(Az_inv_input)
            result = left_multiply_Z(result[0:self.n], self.bm_minimizer.phi_indices, self.n)
            result = np.dot(self.get('rank_m_matrix_inverse'), result)
            result = left_multiply_Z(result, self.bm_minimizer.phi_indices, self.n, transpose=True)
            result = self.left_multiply_hess_barrier_fcn_inv(np.concatenate((result, np.zeros(self.n))))
            return Az_inv_input - result
        else:
            return self.left_multiply_hess_barrier_fcn_inv(input)



