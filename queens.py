import csv
import numpy as np


class Queens:
    def __init__(self, temperature=1.0, cooling_rate=0.95):
        self.n = 8
        self.solution = np.zeros(self.n, dtype=int)
        self.temperature = temperature
        self.cooling_rate = cooling_rate
        self.current_score = None

    def evaluate(self, x):
        attacking_pairs = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if x[i] == x[j] or abs(x[i] - x[j]) == abs(i - j):  # se mesma linha ou na mesma diagonal
                    attacking_pairs += 1
        return attacking_pairs

    def perturb(self, x):
        i, j = np.random.choice(self.n, size=2, replace=False)
        x[i], x[j] = x[j], x[i]
        return x

    def criterion(self, x):
        score = self.evaluate(x)
        if score < self.current_score:
            return True
        else:
            prob = np.exp((self.current_score - score) / self.temperature)
            return np.random.rand() < prob

    def run(self, verbose=False):
        current = np.random.permutation(np.arange(1, 9))
        self.current_score = self.evaluate(current)
        i = 0
        while self.current_score > 0:
            next_solution = self.perturb(current.copy())
            next_score = self.evaluate(next_solution)
            if self.criterion(next_solution):
                self.temperature *= self.cooling_rate
                current, self.current_score = next_solution, next_score
            if i % 100 == 0 and verbose:
                print(f"Iteration {i}: score = {self.current_score}, solution = {current}")
            i += 1
        self.solution = current
        return self.solution, self.evaluate(self.solution), i


class QueensSolver:
    def __init__(self, temperature=1.0, cooling_rate=0.95):
        self.solutions = []
        self.temperature = temperature
        self.cooling_rate = cooling_rate
        self.iters = []

    def solve(self, verbose=False):
        while len(self.solutions) < 92:
            queens = Queens(temperature=self.temperature, cooling_rate=self.cooling_rate)
            solution, score, iters = queens.run()
            if score == 0 and not any(np.array_equal(solution, existing_solution) for existing_solution in self.solutions):
                self.solutions.append(solution)
                self.iters.append(iters)
                if verbose:
                    print(f"Found solution: {solution}, score: {score}, iterations: {iters}")


def find_optimal_parameters(verbose=False):
    data = {
        "temperature": [],
        "cooling_rate": [],
        "mean_iterations": [],
        "std_iterations": [],
        "max_iterations": [],
        "min_iterations": [],
    }

    for i in range(10, 101, 10):
        for j in range(10, 100, 10):
            if verbose:
                print(f"Testing with temperature: {i / 100.0}, cooling rate: {j / 100.0}")
            queens = QueensSolver(temperature=i / 100.0, cooling_rate=j / 100.0)
            queens.solve()
            mean_iters = np.mean(queens.iters)
            data["temperature"].append(i / 100.0)
            data["cooling_rate"].append(j / 100.0)
            data["mean_iterations"].append(mean_iters)
            data["std_iterations"].append(np.std(queens.iters))
            data["max_iterations"].append(np.max(queens.iters))
            data["min_iterations"].append(np.min(queens.iters))
    return data


if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    data = find_optimal_parameters()

    with open("queens_results.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Temperature", "Cooling Rate", "Mean Iterations", "Std Iterations", "Max Iterations", "Min Iterations"])
        for temp, rate, mean_iter, std, _max, _min in zip(data["temperature"], data["cooling_rate"], data["mean_iterations"], data["std_iterations"], data["max_iterations"], data["min_iterations"]):
            writer.writerow([temp, rate, mean_iter, std, _max, _min])
    print("Results saved to queens_results.csv")
