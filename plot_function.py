from matplotlib import pyplot as plt
import numpy as np


def plot_function(function, x1_range, x2_range, num_points=100, alpha=1.0):
    x1 = np.linspace(x1_range[0], x1_range[1], num_points)
    x2 = np.linspace(x2_range[0], x2_range[1], num_points)
    X1, X2 = np.meshgrid(x1, x2)
    X = np.array([X1, X2])
    Z = function(X)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, Z, cmap='jet', edgecolor='none', alpha=alpha)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$f(x_1, x_2)$')
    return fig, ax
