import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from tests.test_Minimizer import generate_data
from src.objects.admm_minimizer import ADMMMinimizer
from src.functions.matrices import left_multiply_Z
from numpy.linalg import norm
from tests.test_BarrierMethodMinimizer import true_solution
import timeit

class ADMMIterationData():
    def __init__(self, admm_minimizer):
        self.vals = {}
        self.vals['x'] = admm_minimizer.x
        self.vals['z'] = admm_minimizer.z
        self.vals['u'] = admm_minimizer.u
        self.vals['objective'] = admm_minimizer.objective(admm_minimizer.x)
        self.vals['primal_epsilon'] = admm_minimizer.primal_epsilon()
        self.vals['dual_epsilon'] = admm_minimizer.dual_epsilon()
        self.vals['primal_residual'] = norm(admm_minimizer.primal_residual(), 2)
        self.vals['dual_residual'] = norm(admm_minimizer.dual_residual(), 2)

m = 64
n = 512

def setUpADMMMinimizer(rho, ratio_of_max=0.1, n_val=n, m_val=m):
    cp, weights, initial_guess, y, phi_indices = generate_data(m_val, n_val)
    admm_lambda_max = np.max(np.absolute(left_multiply_Z(y, phi_indices, n_val, transpose=True)))
    cp.rho = rho
    return ADMMMinimizer(cp, weights, ratio_of_max * admm_lambda_max, y, phi_indices)

def count_iterations(rho):
    N = 20
    total = 0
    for i in range(N):
        minimizer = setUpADMMMinimizer(rho)
        minimizer.solve()
        total += minimizer.iterations
    return total / N

def iterations_vs_rho():
    rho_vals = np.logspace(-1, 1)
    iter_counts = np.zeros(len(rho_vals))
    for i in range(len(rho_vals)):
        iter_counts[i] = count_iterations(rho_vals[i])

    plt.plot(rho_vals, iter_counts, color="k")
    plt.title('Average Iterations over 10 Runs vs. Rho')
    plt.xlabel(r'$\rho$')
    plt.ylabel(r'Iterations')
    plt.show()

def create_history_list(rho):
    minimizer = setUpADMMMinimizer(rho)
    history = [ADMMIterationData(minimizer)]
    while not minimizer.optimal:
        minimizer.iterate()
        history.append(ADMMIterationData(minimizer))
    return history

def get(iter, key):
    return iter.vals[key]

def residuals_vs_iteration(rho):
    history = create_history_list(rho)
    iters = np.linspace(1, len(history), len(history))
    primal_residuals = map(lambda x: get(x, 'primal_residual'), history)
    dual_residuals = map(lambda x: get(x, 'dual_residual'), history)
    primal_epsilon = map(lambda x: get(x, 'primal_epsilon'), history)
    dual_epsilon = map(lambda x: get(x, 'dual_epsilon'), history)

    # plt.subplot(211)
    plt.semilogy(iters, primal_residuals, color="k")
    plt.semilogy(iters, primal_epsilon, linestyle='--', color="k")
    plt.semilogy(iters, dual_residuals, color="b")
    plt.semilogy(iters, dual_epsilon, linestyle='--', color="b")
    plt.title(r'Dual \& Primal Residual and Tolerance for $\rho = ' + str(rho) + r'$ and $\lambda = 0.1 \lambda_{max}$', fontsize=30)
    plt.legend([r'$\|r\|_2$', r'$\epsilon_r$', r'$\|s\|_2$', r'$\epsilon_d$'])
    # plt.ylim([0, 10**-1])
    plt.xlim([1, len(history)])

    # plt.subplot(212)
    # plt.semilogy(iters, dual_residuals, color="k")
    # plt.semilogy(iters, dual_epsilon, linestyle=':', color="k")
    plt.xlabel('Iteration')
    # plt.legend([r'$\|s\|_2$', r'$\epsilon_d$'])
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
        minimizer = setUpADMMMinimizer(normalized_lambda)
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
            minimizer = setUpADMMMinimizer(rho=0.1, ratio_of_max=0.1, n_val=int(n_vals[i][j]), m_val=int(m_vals[i][j]))
            t, result = time(minimizer.solve)
            xstar_cvx, pstar_cvx = true_solution(minimizer)
            time_vals[i][j] = t
            dev_vals[i][j] = norm(xstar_cvx - result[0], 2) / np.sqrt(minimizer.n)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(m_vals, n_vals, time_vals * 1000.0, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_xlabel(r'$m$')
    ax.set_ylabel(r'$n$')
    ax.set_zlabel(r'Computation Time [ms]')
    plt.title('Computation Time vs Problem Dimensions for ADMM', fontsize=30)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(m_vals, n_vals, dev_vals, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_xlabel(r'$m$')
    ax.set_ylabel(r'$n$')
    ax.set_zlabel(r'$\frac{1}{\sqrt{n}} \|x^* - x^{CVX}\|_2$')
    plt.title('2-Norm Accuracy vs Problem Dimensions for ADMM', fontsize=30)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


if __name__=="__main__":
    plt.rc('text', usetex=True)
    plt.rc('font',**{'family':'serif','serif':['Palatino'],'size' : 20})
    # iterations_vs_rho()
    residuals_vs_iteration(0.1)
    # time_deviations_vs_size()