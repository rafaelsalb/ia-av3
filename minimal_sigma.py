import numpy as np

from global_random_search import GlobalRandomSearch
from hill_climb import HillClimb
from local_random_search import LocalRandomSearch


class MinimalSigma:
    def __init__(self, X, limits=None, f=None, minimize=True, optimal_value=0.0):
        self.X = X
        self.limits = limits if limits is not None else np.array([[np.min(X[:, 0]), np.max(X[:, 0])], [np.min(X[:, 1]), np.max(X[:, 1])]])
        self.f = f
        self.minimize = minimize
        self.optimal_value = optimal_value

    def run(self, search_method):
        sigmas = [0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
        passed = []
        for sigma in sigmas:
            for _ in range(100):
                search = search_method(self.X, neighborhood_size=sigma, limits=self.limits, f=self.f, minimize=self.minimize)
                search.optimize()
                if search_method is HillClimb:
                    search.best_value = search.current_score
                if np.all(np.isclose(search.best_value, self.optimal_value, atol=0.01)):
                    passed.append(sigma)
                    break
        return np.min(passed) if passed else None

    def run_all_methods(self):
        results = {
            "hill_climb": self.run(HillClimb),
            "local_random_search": self.run(LocalRandomSearch),
            "global_random_search": self.run(GlobalRandomSearch)
        }
        return results
