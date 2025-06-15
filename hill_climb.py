import numpy as np


class HillClimb:
    def __init__(self, X, max_iter=1000, neighborhood_size=0.1, limits=None, f=None, minimize=True):
        self.X = X
        self.f = f
        self.neighborhood_size = neighborhood_size
        self.limits = limits
        self.max_iters = max_iter
        self.minimize = minimize
        self.current = None
        self.current_score = None
        self.iters = 0

    def _random_solution(self, dim):
        if self.limits is not None:
            return np.random.uniform(self.limits[:, 0], self.limits[:, 1])
        else:
            return np.random.randn(dim)  # unbounded domain

    def _perturb(self, x):
        direction = np.random.randn(*x.shape)
        direction /= np.linalg.norm(direction)
        step = direction * np.random.uniform(0, self.neighborhood_size)
        x_new = x + step
        if self.limits is not None:
            x_new = np.clip(x_new, self.limits[:, 0], self.limits[:, 1])
        return x_new

    def optimize(self, dim=2, verbose=False):
        self.current = self._random_solution(dim)
        self.current_score = self.f(self.current)
        history = []
        rounds_without_improvement = 0

        for i in range(self.max_iters):
            self.iters += 1
            candidate = self._perturb(self.current)
            candidate_score = self.f(candidate)

            is_better = candidate_score < self.current_score if self.minimize else candidate_score > self.current_score
            if is_better:
                self.current, self.current_score = candidate, candidate_score
                history.append((candidate, candidate_score))
                rounds_without_improvement = 0
            else:
                rounds_without_improvement += 1
            if rounds_without_improvement >= 50:  # Stop if no improvement for 50 rounds
                break

            if verbose and (i % 100 == 0 or i == self.max_iters - 1):
                print(f"Iter {i}: score = {self.current_score:.4f}, x = {self.current}")

        return self.current, self.current_score, history
