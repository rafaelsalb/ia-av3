import numpy as np


class MinimalSigma:
    def __init__(self, X, limits=None, f=None, minimize=True, optimal_value=0.0):
        self.X = X
        self.limits = limits if limits is not None else np.array([[np.min(X[:, 0]), np.max(X[:, 0])], [np.min(X[:, 1]), np.max(X[:, 1])]])
        self.f = f
        self.minimize = minimize

    def run(self, search_method):
        sigmas = [0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
        passed = []
        for sigma in sigmas:
            search = search_method(self.X, max_iter=self.max_iter, neighborhood_size=sigma, limits=self.limits, f=self.f, minimize=self.minimize)
            search.optimize()
            if np.approx_equal(search.best_value, self.optimal_value, atol=1e-5):
                passed.append(sigma)
        return np.min(passed) if passed else None
