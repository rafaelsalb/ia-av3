import numpy as np


class LocalRandomSearch:
    def __init__(self, X=None, max_iter=100, neighborhood_size=0.1, limits=None, f=None, minimize=True, initial_x=None):
        self.X = X
        self.max_iter = max_iter
        self.history = []
        self.neighborhood_size = neighborhood_size
        self.limits = limits if limits is not None else np.array([[np.min(X[:, 0]), np.max(X[:, 0])], [np.min(X[:, 1]), np.max(X[:, 1])]])
        self.f = f
        self.minimize = minimize
        if initial_x is not None:
            self.best_x = initial_x
        else:
            self.best_x = self.X[np.random.randint(0, len(self.X))]
        self.best_value = self.f(self.best_x)
        self.iters = 0


    def perturb(self, x):
        # aqui, neighborhood_size é o desvio padrão da distribuição normal
        perturbed_x = x + np.random.normal(0, self.neighborhood_size, size=(2,))
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
        rounds_without_improvement = 0
        for _ in range(self.max_iter):
            self.iters += 1
            perturbed_x = self.perturb(self.best_x)
            perturbed_value = self.f(perturbed_x)

            if (self.minimize and perturbed_value < self.best_value) or (not self.minimize and perturbed_value > self.best_value):
                self.best_x = perturbed_x
                self.best_value = perturbed_value
                self.history.append((self.best_x, self.best_value))
                rounds_without_improvement = 0
            else:
                rounds_without_improvement += 1
            if rounds_without_improvement >= 50:  # Stop if no improvement for 50 rounds
                break

        return self.best_x, self.best_value, self.history

