import numpy as np


class LocalRandomSearch:
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


    def perturb(self, x):
        perturbed_x = x + np.random.uniform(-self.neighborhood_size, self.neighborhood_size, size=x.shape)
        # Ensure perturbed_x is within the limits
        if self.limits is not None:
            if perturbed_x[0] < self.limits[0, 0]:
                perturbed_x[0] = self.limits[0, 0]
            elif perturbed_x[0] > self.limits[0, 1]:
                perturbed_x[0] = self.limits[0, 1]
            if perturbed_x[1] < self.limits[1, 0]:
                perturbed_x[1] = self.limits[1, 0]
            elif perturbed_x[1] > self.limits[1, 1]:
                perturbed_x[1] = self.limits[1, 1]
        return perturbed_x


    def optimize(self):
        self.best_x = self.X[np.random.randint(0, len(self.X))]
        self.best_value = self.f(self.best_x)
        # self.history.append((self.best_x, self.best_value))

        for _ in range(self.max_iter):
            perturbed_x = self.perturb(self.best_x)
            perturbed_value = self.f(perturbed_x)

            if (self.minimize and perturbed_value < self.best_value) or (not self.minimize and perturbed_value > self.best_value):
                self.best_x = perturbed_x
                self.best_value = perturbed_value

            # self.history.append((self.best_x, self.best_value))

        return self.best_x, self.best_value, self.history

