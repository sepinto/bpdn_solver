import numpy as np
import cvxpy as cvx

from src.functions.matrices import *
from src.functions.dct import IDCT
from src.functions.dft import DFT
from src.functions.psychoac import CalcMaskingThreshold, Intensity

class CSReconstructor():
    def __init__(self, coding_params):
        np.random.seed(coding_params.seed)
        self.phi_indices = RandomSubset(coding_params.full_block_length, coding_params.compressed_block_length)

        #TODO: Control these in params?
        self.delta = coding_params.delta
        self.fidelity = coding_params.fidelity
        self.gamma = coding_params.gamma
        self.max_iterations = coding_params.max_iterations
        self.fs = coding_params.sample_rate

        self.full_block_length = coding_params.full_block_length

        self.Z = np.dot(create_Phi(self.full_block_length, self.phi_indices), create_Psi(self.full_block_length))

    def reconstruct(self, y):
        iter_count = 0
        prev_xhat = np.inf * np.ones(self .full_block_length)
        w = np.ones(self.full_block_length)
        while True:
            xhat = self.solve_weighted_l1_problem(w, y)
            iter_count += 1
            if self.check_convergence(prev_xhat, xhat) or iter_count >= self.max_iterations:
                return xhat
            w = self.update_weights(xhat)
            prev_xhat = xhat

    def solve_weighted_l1_problem(self, weights, y):
        xhat = cvx.Variable(self.full_block_length)
        # constraints = [cvx.norm(y - self.Z * xhat) <= self.fidelity]
        constraints = [y == self.Z * xhat]
        objective = cvx.Minimize(cvx.norm(cvx.mul_elemwise(weights, xhat), 1))
        problem = cvx.Problem(objective, constraints)
        problem.solve()
        return np.array(xhat.value.transpose())[0]  # Annoyances in dealing with matrices vs. vectors

    def update_weights(self, xhat):
        return np.divide(1.0, np.abs(xhat) + self.delta)

    def check_convergence(self, prev_xhat, xhat):
        return np.linalg.norm(prev_xhat - xhat) <= self.gamma
