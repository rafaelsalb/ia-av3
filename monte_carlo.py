import matplotlib.pyplot as plt
import numpy as np

from global_random_search import GlobalRandomSearch
from hill_climb import HillClimb
from local_random_search import LocalRandomSearch
from plot_function import plot_function
from utils import mode


class MonteCarlo:
    @staticmethod
    def run(X, max_iter=100, neighborhood_size=0.1, limits=None, f=None, minimize=True, rounds=500):
        """
        Run the Monte Carlo search method.

        :param search_method: The search method to use (e.g., GlobalRandomSearch).
        :param X: Initial points for the search.
        :param max_iter: Maximum number of iterations.
        :param neighborhood_size: Size of the neighborhood for candidate generation.
        :param limits: Limits for the search space.
        :param f: Objective function to optimize.
        :param minimize: Whether to minimize or maximize the objective function.
        :return: Best solution found and its value.
        """

        if f is None:
            raise ValueError("Função objetivo 'f' não pode ser None. Por favor, forneça uma função válida.")

        solution_figs = []
        hist_figs = []
        modes = []
        for i, search_method in enumerate([LocalRandomSearch, GlobalRandomSearch, HillClimb]):
            xs = []
            ys = []
            for _ in range(rounds):
                solver = search_method(X, max_iter=max_iter, neighborhood_size=neighborhood_size, limits=limits, f=f, minimize=minimize)
                best_x, best_value, history = solver.optimize()
                xs.append(best_x)
                ys.append(best_value)
            fig, ax = plot_function(f, x1_range=(limits[0, 0], limits[0, 1]), x2_range=(limits[1, 0], limits[1, 1]), alpha=0.5)
            xs = np.array(xs)
            ax.scatter(xs[:, 0], xs[:, 1], ys, color='g', label='Best Solutions', s=25, alpha=0.7)
            ax.set_title(f"Soluções encontradas via {search_method.__name__} em {rounds} rodadas de Monte Carlo")
            fig.tight_layout()
            solution_figs.append(fig)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title(f"Histograma do valor otimizado - {search_method.__name__}")
            ax.hist(ys, bins=30, alpha=0.7, label=search_method.__name__)
            ax.set_xlabel('Objective Value')
            ax.set_ylabel('Frequency')
            hist_figs.append(fig)

            ys = np.array(ys)
            modes.append(mode(ys))

        return best_x, best_value, ys, modes, solution_figs, hist_figs
