import numpy as np
from local_random_search import LocalRandomSearch
from functions import CostObjective


def main():
    x = np.linspace(-100, 100, 1000)
    x = np.array(np.meshgrid(x, x)).T.reshape(-1, 2)
    x = np.array(x, dtype=np.float64)

    # Create the cost objective function
    f = CostObjective.f_1
    # f = lambda xi: xi[0] + xi[1]

    # Create the local random search optimizer
    optimizer = LocalRandomSearch(X=x, max_iter=1_000_000, neighborhood_size=0.1)

    # Optimize the cost function
    best_x, best_value, history = optimizer.optimize(f)

    # Print the results
    print("Best solution:", best_x)
    print("Best value:", best_value)
    # print("History of solutions:", history)


if __name__ == "__main__":
    main()