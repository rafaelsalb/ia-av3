import numpy as np


class CostObjective:
    @staticmethod
    def f_1(x: np.ndarray):
        """
        Encontre o mínimo
        x: np.ndarray ∈ [-100, 100]
        """
        return x[0] ** 2 + x[1] ** 2

    @staticmethod
    def f_2(x: np.ndarray):
        """
        Encontre o máximo
        x[0] ∈ [-2, 4]
        x[1] ∈ [-2, 5]
        """
        return np.exp(-(x[0] ** 2 + x[1] ** 2)) + (2 * np.exp(-((x[0] - 1.7) ** 2 + (x[1] - 1.7) ** 2)))

    @staticmethod
    def f_3(x: np.ndarray):
        """
        Encontre o mínimo
        x: np.ndarray ∈ [-8, 8]
        """
        return -20 * np.exp(-(0.2 * np.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2)))) - np.exp(
            0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))) + np.e + 20

    @staticmethod
    def f_4(x: np.ndarray):
        """
        Encontre o mínimo
        x: np.ndarray ∈ [-5.12, 5.12]
        """
        return (x[0] ** 2 - 10 * np.cos(2 * np.pi * x[0]) + 10) + (x[1] ** 2 - 10 * np.cos(2 * np.pi * x[1]) + 10)

    @staticmethod
    def f_5(x: np.ndarray):
        """
        Encontre o mínimo
        x: np.ndarray ∈ [-10, 10]
        """
        return (x[0] * np.cos(x[0]) / 20) + 2 * np.exp(-((x[0] ** 2) -((x[1] - 1) ** 2))) + 0.01 * x[0] * x[1]

    @staticmethod
    def f_6(x: np.ndarray):
        """
        Encontre o mínimo
        x: np.ndarray ∈ [-1, 3]
        """
        return x[0] * np.sin(4 * np.pi * x[0]) - x[1] * np.sin(4 * np.pi * x[1] + np.pi) + 1

    @staticmethod
    def f_7(x: np.ndarray):
        """
        Encontre o mínimo
        x: np.ndarray ∈ [0, π]
        """
        return -np.sin(x[0]) * np.sin(x[0] ** 2 / np.pi) - np.sin(x[1]) * np.sin(2 * x[1] ** 2 / np.pi)

    @staticmethod
    def f_8(x: np.ndarray):
        """
        Encontre o mínimo
        x: np.ndarray ∈ [-200, 20]
        """
        return -(x[1] + 47) * np.sin(np.sqrt(abs(x[1] + (x[0] / 2) + 47))) - x[0] * np.sin(np.sqrt(abs(x[0] - (x[1] + 47))))
