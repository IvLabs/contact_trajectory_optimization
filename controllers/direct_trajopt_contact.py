""" Trajectory Optimization formulation as given in
- A Direct Method for Trajectory Optimization of Rigid Bodies Through Contact """

import casadi as ca
import numpy as np
from contact_trajectory_optimization.envs.finger_contact import FingerContact


class NLP:
    def __init__(self):
        super().__init__()
        self.model = FingerContact()
        self.__setOptimizationParams__(total_duration=2, n_steps=20)

        self.opti = ca.Opti()

        self.__setOptimizationParams__()
        self.__setVariables__()
        self.__setConstraints__()

    def __setOptimizationParams__(self, total_duration=2, n_steps=20):
        self.T = total_duration
        self.N = n_steps

    def __setVariables__(self):
        self.h = self.opti.variable(1, 1)
        self.states = []
        self.dstates = []
        self.actions = []
        self.forces = []

        for i in range(self.N+1):
            self.states.append(self.opti.variable(self.model.n_joints, 1))
            self.dstates.append(self.opti.variable(self.model.n_joints, 1))
            if i > 0:
                self.actions.append(self.opti.variable(self.model.dof, 1))
                self.forces.append(self.opti.variable(self.model.dims, self.model.n_contact))

    def __setConstraints__(self):
        for k in range(1, self.N):
            q_1, dq_1 = self.states[k], self.dstates[k]
            q_2, dq_2 = self.states[k+1], self.dstates[k+1]
            u_1, u_2 = self.actions[k], self.actions[k + 1]
            lam_1, lam_2 = self.forces[k], self.forces[k+1]

            # dynamics constraint
            f_1 = self.model.dynamics(q=q_1, dq=dq_1, lam=lam_1)
            f_2 = self.model.dynamics(q=q_2, dq=dq_2, lam=lam_2)

            self.opti.subject_to(q_1 - q_2 + self.h*dq_2 == 0)
            self.opti.subject_to(f_2['H'] @ (dq_2 - dq_1) + self.h * f_2['C'] @ (q_2 - q_1) - (f_2['B'] @ u_2 + f_2['J_phi'].T @ lam_2) == 0)

Problem = NLP()