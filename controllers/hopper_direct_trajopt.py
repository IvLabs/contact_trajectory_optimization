""" Trajectory Optimization formulation as given in
- A Direct Method for Trajectory Optimization of Rigid Bodies Through Contact """

import casadi as ca
import numpy as np
from contact_trajectory_optimization.envs.hopper import Hopper
from matplotlib import pyplot as plt


class NLP:
    def __init__(self):
        super().__init__()
        self.model = Hopper()
        self.__setOptimizationParams__(total_duration=2, n_steps=20, epsilon=1e-4)

        self.opti = ca.Opti()
        self.var_dt = True

        # self.__setVariables__()
        # self.__setConstraints__()
        # self.__setCosts__()

    def __setOptimizationParams__(self, total_duration, n_steps, epsilon):
        self.T = total_duration
        self.N = n_steps
        self.epsilon = epsilon
