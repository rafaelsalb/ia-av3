import numpy as np

from hill_climb import HillClimb
from utils import mode


class MonteCarlo:
    @staticmethod
    def run(search_method, X, max_iter=100, neighborhood_size=0.1, limits=None, f=None, minimize=True, rounds=500):
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

        xs = []
        ys = []
        for _ in range(rounds):
            solver = search_method(X, max_iter=max_iter, neighborhood_size=neighborhood_size, limits=limits, f=f, minimize=minimize)
            best_x, best_value, history = solver.optimize()
            xs.append(best_x)
            ys.append(best_value)

        ys = np.array(ys)
        most_common = mode(ys)

        return best_x, best_value, ys, most_common
