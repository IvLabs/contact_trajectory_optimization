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

        self.__setVariables__()
        # self.__setConstraints__()
        # self.__setCosts__()

    def __setOptimizationParams__(self, total_duration, n_steps, epsilon):
        self.T = total_duration
        self.N = n_steps
        self.epsilon = epsilon

    def __setVariables__(self):
        if self.var_dt:
            self.h = self.opti.variable(self.N - 1)
            self.opti.subject_to(self.opti.bounded(0, self.h, self.model.dt))
            self.opti.subject_to(ca.sum1(self.h) == self.T / self.N)
            self.opti.set_initial(self.h, [0.05]*(self.N-1))
        else:
            self.h = 0.05

        # self.states = []
        # self.dstates = []
        # self.actions = []
        # self.forces, self.gammas = [], []
        #
        # for i in range(self.N):
        #     self.states.append(self.opti.variable(self.model.n_generalized, 1))
        #     self.dstates.append(self.opti.variable(self.model.n_generalized, 1))
        #
        #     self.actions.append(self.opti.variable(self.model.dof, 1))
        #     self.forces.append(self.opti.variable(self.model.dims + 1, self.model.n_contact))
        #     self.gammas.append(self.opti.variable(1))

        self.states = self.opti.variable(self.model.n_generalized, self.N)
        self.dstates = self.opti.variable(self.model.n_generalized, self.N)
        self.actions = self.opti.variable(self.model.dof, self.N)
        self.forces, self.gammas = self.opti.variable(3, self.N), self.opti.variable(1, self.N)

    def __setConstraints__(self):
        self.opti.subject_to(self.states[:, 0] == [0, (self.model.length[1]*np.cos(np.pi/3) +
                                                       self.model.length[2]*np.cos(np.pi/1.5)),
                                                   np.pi / 3, np.pi / 1.5])
        self.opti.subject_to(self.states[:, -1] == [2, (self.model.length[1]*np.cos(np.pi/3) +
                                                       self.model.length[2]*np.cos(np.pi/1.5)),
                                                   np.pi / 3, np.pi / 1.5])

        self.opti.subject_to(self.dstates[:, 0] == [0] * self.model.n_generalized)
        self.opti.subject_to(self.dstates[:, -1] == [0] * self.model.n_generalized)

        k = 0

        for k in range(self.N - 1):
            q_1, dq_1 = self.states[k], self.dstates[k]
            q_2, dq_2 = self.states[k + 1], self.dstates[k + 1]
            u_1, u_2 = self.actions[k], self.actions[k + 1]

            lam_1, lam_2 = self.forces[k], self.forces[k + 1]

            f_1 = self.model.dynamics(q=q_1, dq=dq_1, lam=lam_1)
            f_2 = self.model.dynamics(q=q_2, dq=dq_2, lam=lam_2)


problem = NLP()
