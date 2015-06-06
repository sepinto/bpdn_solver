import numpy as np
import cvxpy as cvx
from src.functions.matrices import create_Z, left_multiply_Z
from numpy.linalg import norm

class Minimizer():
    def __init__(self, coding_params, weights, initial_guess, lambd=0, y=None, phi_indices=None):
        self.coding_params = coding_params
        self.n = int(coding_params.full_block_length)
        self.m = int(coding_params.compressed_block_length)
        self.weights = np.copy(weights)
        self.initial_guess = np.copy(initial_guess)
        self.lambd = lambd
        self.y = np.copy(y)
        self.phi_indices = np.copy(phi_indices)

        # Values we'll use for testing so calculate on demand
        self.vals = {}

    def get(self, key):
        if key in self.vals:
            return self.vals[key]
        else:
            val = getattr(self, 'calc_' + key)()
            self.vals[key] = np.copy(val)
            return val

    def calc_Z(self):
        return create_Z(self.phi_indices, self.n)

    def objective(self, x):
        return norm(self.weights * x, 1) + self.lambd * norm(self.y - left_multiply_Z(x, self.phi_indices, self.n), 2)**2

    def cvx_objective(self, x):
        return cvx.norm(np.diag(self.weights) * x, 1) + self.lambd * cvx.norm(self.y - self.get('Z') * x)**2.0

    def solve(self):
        xhat = cvx.Variable(self.n)
        objective = cvx.Minimize(self.cvx_objective(xhat))
        problem = cvx.Problem(objective)
        pstar = problem.solve()
        return np.array(xhat.value.transpose())[0], pstar

