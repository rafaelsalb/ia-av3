import numpy as np


class GlobalRandomSearch:
    def __init__(self, X, max_iter=100, neighborhood_size=0.1, limits=None, f=None, minimize=True):
        self.X = X
        self.max_iter = max_iter
        self.best_x = None
        self.best_value = None
        self.history = []
        self.neighborhood_size = neighborhood_size
        self.limits = limits if limits is not None else np.array([[np.min(X[:, 0]), np.max(X[:, 0])], [np.min(X[:, 1]), np.max(X[:, 1])]])
        self.f = f
        self.minimize = minimize


    def candidate(self):
        cand = np.random.normal(0, self.neighborhood_size, size=(2,))
        # Ensure cand is within the limits
        if self.limits is not None:
            if cand[0] < self.limits[0, 0]:
                cand[0] = self.limits[0, 0]
            elif cand[0] > self.limits[0, 1]:
                cand[0] = self.limits[0, 1]
            if cand[1] < self.limits[1, 0]:
                cand[1] = self.limits[1, 0]
            elif cand[1] > self.limits[1, 1]:
                cand[1] = self.limits[1, 1]
        return cand


    def optimize(self):
        self.best_x = self.X[0]
        self.best_value = self.f(self.best_x)
        self.history.append((self.best_x, self.best_value))
        without_improvement = 0

        for _ in range(self.max_iter):
            x_cand = self.candidate()
            perturbed_value = self.f(x_cand)

            if (self.minimize and perturbed_value < self.best_value) or (not self.minimize and perturbed_value > self.best_value):
                self.best_x = x_cand
                self.best_value = perturbed_value
                self.history.append((self.best_x, self.best_value))
                without_improvement = 0
            else:
                without_improvement += 1
            if without_improvement >= 10:  # Stop if no improvement for 10 rounds
                break

        return self.best_x, self.best_value, self.history

