import numpy as np
from src.objects.newton_iteration_data import NewtonIterationData
from src.functions.matrices import left_multiply_Z, create_Z
from src.objects.minimizer import Minimizer

class BarrierMethodMinimizer(Minimizer):
    def __init__(self, coding_params, weights, initial_guess, initial_t, lambd=0, y=None, phi_indices=None):
        # Set parameters
        Minimizer.__init__(self, coding_params, weights, initial_guess, lambd=lambd, y=y, phi_indices=phi_indices)
        self.initial_t = initial_t
        self.quadratic = self.lambd > 0

        # Calculate these once and never again
        self.num_inequalities = 2.0 * self.n
        self.norm_y = np.dot(y, y)
        self.ZTy = left_multiply_Z(y, self.phi_indices, self.n, transpose=True)

        # Iteration variables for barrier method
        self.iterations = 0
        self.optimal = False
        self.optimal_nt = False
        self.t = initial_t
        self.z = np.concatenate((self.initial_guess, 2.0 * self.weights * self.initial_guess * np.sign(self.initial_guess)))
        self.iter_data = NewtonIterationData(self.z, self)

    def calc_A(self):
        return np.bmat(([np.diag(self.weights), -1 * np.eye(self.n)], [-1 * np.diag(self.weights), -1 * np.eye(self.n)]))

    def calc_Z(self):
        return create_Z(self.phi_indices, self.n)

    def solve(self):
        while not self.optimal:
            # self.barrier_step()
            self.iterate()
        x_star = self.iter_data.z[0:self.n]
        return x_star, self.objective(x_star)

    def iterate(self):
        if not self.optimal_nt: # On centering step
            self.newton_iteration()
        else: # Move to next centering step
            self.optimal_nt = False # Reset for next centering step
            self.z = np.copy(self.iter_data.z)
            if (self.num_inequalities / self.t) < self.coding_params.epsilon:
                self.optimal = True
                self.iter_data = NewtonIterationData(self.z, self)
                return
            self.t = self.coding_params.mu * self.t
            self.iter_data = NewtonIterationData(self.z, self)

    def newton_iteration(self):
        z_step = -1.0 * self.iter_data.left_multiply_hess_newton_obj_inv(self.iter_data.get('grad_newton_obj'))
        z_dec = np.dot(self.iter_data.get('grad_newton_obj'), -1.0 * z_step)
        if z_dec <= 2.0 * self.coding_params.epsilon:
            self.optimal_nt = True
            return
        t_nt = self.line_search(z_step)
        self.iter_data = NewtonIterationData(self.iter_data.z + t_nt * z_step, self)
        self.iterations += 1

    def line_search(self, z_step):
        t = 1.0
        while not self.feasible_z(self.iter_data.z + t * z_step): #TODO: set this initially
            t = self.coding_params.beta * t
        while self.newton_obj(self.iter_data.z + t * z_step) >= self.iter_data.get('newton_obj') + self.coding_params.alpha * t * np.dot(self.iter_data.get('grad_newton_obj'), z_step):
            t = self.coding_params.beta * t
        return t

    def one_norm_obj(self, z):
        return sum(z[self.n:])

    def two_norm_obj(self, z, Zx=None):
        x = z[0:self.n]
        Zx = left_multiply_Z(x, self.phi_indices, self.n) if Zx is None else Zx
        val = np.dot(Zx, Zx) - 2.0 * np.dot(self.y, Zx) + self.norm_y
        return val

    def z_objective(self, z, Zx=None):
        return self.one_norm_obj(z) + self.lambd * self.two_norm_obj(z, Zx)

    def linear_inequality_constraint(self, z):
        x = z[0:self.n]
        s = z[self.n:]
        val = np.concatenate((self.weights * x - s, -1.0 * self.weights * x - s))
        return val

    def feasible_z(self, z):
        return np.all(self.linear_inequality_constraint(z) < 0)

    def barrier_fcn(self, z):
        return np.sum(-1.0 * np.log(-1.0 * self.linear_inequality_constraint(z)))

    def newton_obj(self, z):
        return self.t * self.z_objective(z) + self.barrier_fcn(z)




