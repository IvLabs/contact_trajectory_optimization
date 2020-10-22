""" Trajectory Optimization formulation as given in
- A Direct Method for Trajectory Optimization of Rigid Bodies Through Contact """

import casadi as ca
import numpy as np
from contact_trajectory_optimization.envs.finger_contact import FingerContact
from matplotlib import pyplot as plt


class NLP:
    def __init__(self):
        super().__init__()
        self.model = FingerContact()
        self.__setOptimizationParams__(total_duration=2, n_steps=50)

        self.opti = ca.Opti()
        self.var_dt = False
        self.__setVariables__()
        self.__setConstraints__()
        self.__setCosts__()

    def __setOptimizationParams__(self, total_duration, n_steps):
        self.T = total_duration
        self.N = n_steps

    def __setVariables__(self):
        if self.var_dt:
            self.h = self.opti.variable(1)
            self.opti.subject_to(self.h > 0)
            self.opti.subject_to(self.h <= self.model.dt)
            self.opti.set_initial(self.h, 0.05)
        else:
            self.h = 0.05

        self.states = []
        self.dstates = []
        self.actions = []
        self.forces, self.gammas = [], []

        for i in range(self.N):
            self.states.append(self.opti.variable(self.model.n_joints, 1))
            self.dstates.append(self.opti.variable(self.model.n_joints, 1))
            # if i == 0:
            #     self.actions.append(ca.DM.zeros(self.model.dof, 1))
            #     self.forces.append(ca.DM.zeros(self.model.dims, self.model.n_contact))
            # else:
            # if i > 0:
            self.actions.append(self.opti.variable(self.model.dof, 1))
            self.forces.append(self.opti.variable(self.model.dims, self.model.n_contact))
            # self.forces.append(self.opti.variable(self.model.dims + 1, self.model.n_contact))
            # self.gammas.append(self.opti.variable(1))

    def __setConstraints__(self):
        self.opti.subject_to(self.states[0] == [np.pi / 3, 0, 0])
        self.opti.subject_to(self.dstates[0] == [0]*3)
        self.opti.subject_to(self.dstates[-1] < -0.5)
        self.phis = ca.MX.zeros(2, self.N)
        for k in range(self.N - 1):
            q_1, dq_1 = self.states[k], self.dstates[k]
            q_2, dq_2 = self.states[k + 1], self.dstates[k + 1]
            u_1, u_2 = self.actions[k], self.actions[k + 1]

            lam_1, lam_2 = self.forces[k], self.forces[k+1]

            # lam_1, lam_2 = ca.MX.zeros(2, 1), ca.MX.zeros(2, 1)
            # lam_1[0, 0], lam_1[1, 0] = self.forces[k][0, 0] - self.forces[k][1, 0], self.forces[k][2, 0]
            # lam_2[0, 0], lam_2[1, 0] = self.forces[k + 1][0, 0] - self.forces[k + 1][1, 0], self.forces[k + 1][2, 0]

            kine_1 = self.model.kinematics(q=q_1, dq=dq_1)
            kine_2 = self.model.kinematics(q=q_2, dq=dq_2)

            # dynamics constraint
            f_1 = self.model.dynamics(q=q_1, dq=dq_1, lam=lam_1)
            f_2 = self.model.dynamics(q=q_2, dq=dq_2, lam=lam_2)

            self.opti.subject_to(q_1 - q_2 + self.h * dq_2 == 0)
            self.opti.subject_to(f_2['H'] @ (dq_2 - dq_1) +
                                 self.h * (f_2['C'] @ dq_2 + f_2['G'] - f_2['B'] @ u_2 - f_2['J_phi'].T @ lam_2) == 0)
            self.phis[:, k] = kine_1['x'][:, 1] - kine_1['x'][:, 2]

            # friction constraints
            self.opti.subject_to(lam_1 >= 0)
            self.opti.subject_to(f_1['phi'] >= 0)
            self.opti.subject_to(f_1['phi'].T @ lam_1 == 0)

            # lam_1_z, lam_1_xp, lam_1_xm = self.forces[k][2, 0], self.forces[k][0, 0], self.forces[k][1, 0]
            # gam_1 = self.gammas[k]
            # psi_1 = kine_1['dx'][:, 1]
            # self.opti.subject_to(f_1['phi'] >= 0)
            # self.opti.subject_to([lam_1_z >= 0, lam_1_xp >= 0, lam_1_xm >= 0, gam_1 >= 0])
            # self.opti.subject_to(self.model.mu * lam_1_z - lam_1_xp - lam_1_xm >= 0)
            # self.opti.subject_to(gam_1 + psi_1 >= 0)
            # self.opti.subject_to(gam_1 - psi_1 >= 0)
            #
            # self.opti.subject_to(f_1['phi'].T @ lam_1_z == 0)
            # # self.opti.subject_to(f_1['phi'].T @ lam_1_z <= 0)
            #
            # self.opti.subject_to((self.model.mu * lam_1_z - lam_1_xp - lam_1_xm).T @ lam_1_z == 0)
            #
            # self.opti.subject_to((gam_1 + psi_1).T @ lam_1_xp <= 0)
            # self.opti.subject_to((gam_1 + psi_1).T @ lam_1_xp >= 0)
            #
            # self.opti.subject_to((gam_1 - psi_1).T @ lam_1_xm == 0)
            # self.opti.subject_to((gam_1 - psi_1).T @ lam_1_xm >= 0)

            # bounds model specific
            # self.opti.bounded([-np.pi]*3, q_1, [np.pi]*3)
            self.opti.subject_to(q_1[0] <= np.pi)
            self.opti.subject_to(q_1[0] >= 0)
            self.opti.subject_to(q_1[1] < 2*np.pi/3)
            self.opti.subject_to(q_1[1] > -2*np.pi/3)
            # self.opti.subject_to(q_1[0] <= np.pi)
            # self.opti.subject_to(q_1[0] >= -np.pi)
            # self.opti.subject_to(ca.fabs(q_1[0] + q_1[1]) > 0)

    def __setCosts__(self):
        Q = ca.diag(ca.MX([1, 1, 0]))
        R = ca.diag(ca.MX([1, 1]))
        cost = 0
        for k in range(self.N):
            q, dq = self.states[k], self.dstates[k]
            u = self.actions[k]
            cost += dq.T @ Q @ dq + u.T @ R @ u

        self.opti.minimize(cost + ca.sum2(ca.sum1(self.phis**2)))

    def __solve__(self):
        p_opts = {"expand": True}
        s_opts = {"max_iter": 3000}
        self.opti.solver("ipopt", p_opts, s_opts)

        try:
            self.solution = self.opti.solve_limited()
            self.debug = False
        except:
            self.debug = True

    def __interpolate__(self):
        k = 0.005
        self.x1 = []
        self.x2 = []
        self.x3 = []
        self.u1 = []
        self.u2 = []

        if self.debug:
            if self.var_dt:
                self.dt = self.opti.debug.value(self.h)
            else:
                self.dt = self.h

            for i in range(self.N):
                self.x1.append(self.opti.debug.value(self.states[i][0]))
                self.x2.append(self.opti.debug.value(self.states[i][1]))
                self.x3.append(self.opti.debug.value(self.states[i][2]))
        else:
            self.dt = self.solution.value(self.h)
            for i in range(self.N):
                self.x1.append(self.solution.value(self.states[i][0]))
                self.x2.append(self.solution.value(self.states[i][1]))
                self.x3.append(self.solution.value(self.states[i][2]))

        self.__plot__()

    def __plot__(self):
        t = np.linspace(0, self.N * self.dt, self.N)
        plt.plot(t, self.x1, 'o')
        plt.plot(t, self.x2, 'o')
        plt.plot(t, self.x3, 'o')

        plt.show()

        self.model.visualize(self.x1, self.x2, self.x3, t)


Problem = NLP()
Problem.__solve__()
Problem.__interpolate__()
