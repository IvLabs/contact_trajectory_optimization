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
        self.__setCosts__()

    def __setOptimizationParams__(self, total_duration=2, n_steps=20):
        self.T = total_duration
        self.N = n_steps

    def __setVariables__(self):
        self.h = self.opti.variable(1)
        self.opti.subject_to(self.h > 0)
        self.opti.subject_to(self.h <= self.model.dt)

        self.states = []
        self.dstates = []
        self.actions = []
        self.forces, self.gammas = [], []

        for i in range(self.N + 1):
            self.states.append(self.opti.variable(self.model.n_joints, 1))
            self.dstates.append(self.opti.variable(self.model.n_joints, 1))
            # if i == 0:
            #     self.actions.append(ca.DM.zeros(self.model.dof, 1))
            #     self.forces.append(ca.DM.zeros(self.model.dims, self.model.n_contact))
            # else:
            if i > 0:
                self.actions.append(self.opti.variable(self.model.dof, 1))
                self.forces.append(self.opti.variable(self.model.dims + 1, self.model.n_contact))
                self.gammas.append(self.opti.variable(1))

    def __setConstraints__(self):
        for k in range(self.N - 1):
            q_1, dq_1 = self.states[k], self.dstates[k]
            q_2, dq_2 = self.states[k + 1], self.dstates[k + 1]
            u_1, u_2 = self.actions[k], self.actions[k + 1]

            lam_1, lam_2 = ca.MX.zeros(2, 1), ca.MX.zeros(2, 1)
            lam_1[0, 0], lam_1[1, 0] = self.forces[k][0, 0] - self.forces[k][1, 0], self.forces[k][2, 0]
            lam_2[0, 0], lam_2[1, 0] = self.forces[k + 1][0, 0] - self.forces[k + 1][1, 0], self.forces[k + 1][2, 0]

            kine_1 = self.model.kinematics(q=q_1, dq=dq_1)
            kine_2 = self.model.kinematics(q=q_2, dq=dq_2)

            # dynamics constraint
            f_1 = self.model.dynamics(q=q_1, dq=dq_1, lam=lam_1)
            f_2 = self.model.dynamics(q=q_2, dq=dq_2, lam=lam_2)

            self.opti.subject_to(q_1 - q_2 + self.h * dq_2 == 0)
            self.opti.subject_to(f_2['H'] @ (dq_2 - dq_1) + self.h * f_2['C'] @ (q_2 - q_1) - f_2['B'] @ u_2 - f_2[
                'J_phi'].T @ lam_2 == 0)

            # friction constraints
            lam_1_z, lam_1_xp, lam_1_xm = self.forces[k][2, 0], self.forces[k][0, 0], self.forces[k][1, 0]
            gam_1 = self.gammas[k]
            psi_1 = kine_1['dx'][:, 1]
            self.opti.subject_to(f_1['phi'] >= 0)
            self.opti.subject_to([lam_1_z >= 0, lam_1_xp >= 0, lam_1_xm >= 0, gam_1 >= 0])
            self.opti.subject_to(self.model.mu * lam_1_z - lam_1_xp - lam_1_xm >= 0)
            self.opti.subject_to(gam_1 + psi_1 >= 0)
            self.opti.subject_to(gam_1 - psi_1 >= 0)
            self.opti.subject_to(f_1['phi'].T @ lam_1_z == 0)
            self.opti.subject_to((self.model.mu * lam_1_z - lam_1_xp - lam_1_xm).T @ lam_1_z == 0)
            # self.opti.subject_to((gam_1 + psi_1).T @ lam_1_xp == 0)
            # self.opti.subject_to((gam_1 - psi_1).T @ lam_1_xm == 0)

            # bounds model specific
            # self.opti.subject_to(ca.fabs(q_1) <= np.pi)
            # self.opti.subject_to(ca.fabs(q_2) <= np.pi)

    def __setCosts__(self):
        Q = ca.diag(ca.MX([0, 0, 1]))
        R = ca.diag(ca.MX([1, 1]))
        cost = 0
        for k in range(1, self.N):
            q, dq = self.states[k], self.dstates[k]
            u = self.actions[k]
            cost += q.T @ Q @ q + u.T @ R @ u

        self.opti.minimize(cost)

    def __solve__(self):
        p_opts = {"expand": True}
        s_opts = {"max_iter": 3000}
        self.opti.solver("ipopt", p_opts, s_opts)
        self.solution = self.opti.solve()


Problem = NLP()
Problem.__solve__()
