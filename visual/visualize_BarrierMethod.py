import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from tests.test_Minimizer import generate_data
from src.objects.barrier_method_minimizer import BarrierMethodMinimizer
from src.functions.matrices import left_multiply_Z
from numpy.linalg import norm
from tests.test_BarrierMethodMinimizer import true_solution
import timeit
class BMIterationData():
    def __init__(self, bm_minimizer, true_xstar, true_pstar):
        self.vals = {}
        self.vals['x'] = bm_minimizer.iter_data.z[0:bm_minimizer.n]
        self.vals['objective'] = bm_minimizer.objective(self.vals['x'])
        self.vals['pstar_diff'] = np.absolute(self.vals['objective'] - true_pstar)
        self.vals['2norm_x_diff'] = norm(self.vals['x'] - true_xstar, 2)

m = 64
n = 512

def setUpBMMinimizer(normalized_lambda=1.5, n_val=n, m_val=m):
    cp, weights, initial_guess, y, phi_indices = generate_data(m_val, n_val)
    lambda_min = np.max(np.absolute(left_multiply_Z(y, phi_indices, n_val, transpose=True))) #1.0 / (4.0 * np.max(np.absolute(left_multiply_Z(y, phi_indices, n, transpose=True))))
    # (self, coding_params, weights, initial_guess, initial_t, lambd=0, y=None, phi_indices=None):
    return BarrierMethodMinimizer(cp, weights, initial_guess, 1.0, normalized_lambda * lambda_min, y, phi_indices)

def something_vs_lambda(fcn_of_lambda, log_min, log_max):
    lambda_vals = np.logspace(log_min,log_max,10)
    results = []
    for i in range(len(lambda_vals)):
        results.append(fcn_of_lambda(lambda_vals[i]))
    return lambda_vals, results

def xnorm(normalized_lambda=1.5):
    minimizer = setUpBMMinimizer(normalized_lambda)
    xstar_cvx, pstar_cvx = true_solution(minimizer)
    return norm(xstar_cvx, 2)


def iterations(normalized_lambda=1.5):
    N = 10
    total = 0
    for i in range(N):
        minimizer = setUpBMMinimizer(normalized_lambda)
        minimizer.solve()
        total += minimizer.iterations
    return total / N

def deviations(normalized_lambda=1.5):
    minimizer = setUpBMMinimizer(normalized_lambda)
    xstar_cvx, pstar_cvx = true_solution(minimizer)
    xstar_mine, pstar_mine = minimizer.solve()
    return [norm(xstar_cvx - xstar_mine, np.inf), norm(xstar_cvx - xstar_mine, 1) / minimizer.n, norm(xstar_cvx - xstar_mine, 2) / np.sqrt(minimizer.n)]

def iterations_vs_lambda():
    lambd, iters = something_vs_lambda(iterations, 0, 2)
    plt.plot(lambd, iters, color="k")
    plt.title('Avg. Iterations vs Normalized Lambda')
    plt.xlabel('Normalized Lambda')
    plt.ylabel('Average Iterations')
    plt.show()
    # plt.ylim([0, 10**-1])
    # plt.xlim([1, len(iters)])

def deviations_vs_lambda():
    lambd, devs = something_vs_lambda(deviations, 0, 2)
    plt.plot(lambd, map(lambda x: x[0], devs), color="k")
    plt.plot(lambd, map(lambda x: x[1], devs), color="b")
    plt.plot(lambd, map(lambda x: x[2], devs), color="r")
    plt.title('Deviations in x vs Normalized Lambda')
    plt.xlabel('Normalized Lambda')
    plt.ylabel('Deviation in x')
    plt.legend(['Maximum Absolute Deviation', 'Mean Absolute Deviation', 'Standard Deviation'])
    plt.show()
    # plt.ylim([0, 10**-1])
    # plt.xlim([1, len(iters)])

def xnorm_vs_lambda():
    lambd, xnorms = something_vs_lambda(xnorm, -4, -2.3)
    plt.plot(lambd, xnorms, color="k")
    plt.title('||x*||_2 vs Normalized Lambda')
    plt.xlabel('Normalized Lambda')
    plt.ylabel('||x*||_2')
    plt.show()

def create_history_list(lambd):
    minimizer = setUpBMMinimizer(lambd)
    xstar_cvx, pstar_cvx = true_solution(minimizer)
    history = [BMIterationData(minimizer, xstar_cvx, pstar_cvx)]
    while not minimizer.optimal:
        minimizer.iterate()
        history.append(BMIterationData(minimizer, xstar_cvx, pstar_cvx))
    return history

def get(iter, key):
    return iter.vals[key]

def convergence(lambd):
    history = create_history_list(lambd)
    iters = np.linspace(1, len(history), len(history))

    pstar_diff = map(lambda x: get(x, 'pstar_diff'), history)
    twonorm_x_diff = map(lambda x: get(x, '2norm_x_diff'), history)

    plt.subplot(211)
    plt.semilogy(iters, pstar_diff, color="k")
    plt.title(r'Convergence of Barrier Method', fontsize=30)
    plt.ylabel(r'$|p - p^*|$', fontsize=20)
    # plt.ylim([0, 10**-1])
    plt.xlim([1, len(history)])

    plt.subplot(212)
    plt.semilogy(iters, twonorm_x_diff, color="k")
    plt.xlabel(r'Iteration', fontsize=20)
    plt.ylabel(r'$\|x - x^*\|_2$', fontsize=20)
    # plt.ylim([0, 1])
    plt.xlim([1, len(history)])
    plt.show()

def time(fcn, *args):
    t1 = timeit.default_timer()
    result = fcn(*args)
    t2 = timeit.default_timer()
    return t2 - t1, result

def times(normalized_lambda=1.5):
    N = 10
    total = 0
    for i in range(N):
        minimizer = setUpBMMinimizer(normalized_lambda)
        t, result = time(minimizer.solve)
        total += t
    return total / N

def time_deviations_vs_size():
    n_vals = np.logspace(2.1, 3, 10)
    m_vals = np.logspace(1, 2, 10)
    n_vals, m_vals = np.meshgrid(n_vals, m_vals)
    time_vals = np.zeros_like(n_vals)
    dev_vals = np.zeros_like(n_vals)
    for i in range(len(n_vals)):
        for j in range(len(n_vals[i])):
            minimizer = setUpBMMinimizer(normalized_lambda=1.5, n_val=int(n_vals[i][j]), m_val=int(m_vals[i][j]))
            t, result = time(minimizer.solve)
            xstar_cvx, pstar_cvx = true_solution(minimizer)
            time_vals[i][j] = t
            dev_vals[i][j] = norm(xstar_cvx - result[0], 2) / np.sqrt(minimizer.n)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(m_vals, n_vals, time_vals, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_xlabel(r'$m$')
    ax.set_ylabel(r'$n$')
    ax.set_zlabel(r'Computation Time [s]')
    plt.title('Computation Time vs Problem Dimensions for Log Barrier Method', fontsize=30)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(m_vals, n_vals, dev_vals * 1000.0, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_xlabel(r'$m$')
    ax.set_ylabel(r'$n$')
    ax.set_zlabel(r'$\frac{1}{\sqrt{n}} \|x^* - x^{CVX}\|_2$ $\times 10^{-3}$')
    plt.title('2-Norm Accuracy vs Problem Dimensions for Log Barrier Method', fontsize=30)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

if __name__=="__main__":
    plt.rc('text', usetex=True)
    plt.rc('font',**{'family':'serif','serif':['Palatino'],'size' : 20})
    convergence(2)
    # deviations_vs_lambda()
    # iterations_vs_lambda()
    # xnorm_vs_lambda()
    # time_deviations_vs_size()