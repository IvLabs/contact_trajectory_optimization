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
        self.__setOptimizationParams__(total_duration=2, n_steps=20, epsilon=1e-4)

        self.opti = ca.Opti()
        self.var_dt = True
        self.__setVariables__()
        self.__setConstraints__()
        self.__setCosts__()

    def __setOptimizationParams__(self, total_duration, n_steps, epsilon):
        self.T = total_duration
        self.N = n_steps
        self.epsilon = epsilon

    def __setVariables__(self):
        if self.var_dt:
            self.h = self.opti.variable(1)
            self.opti.subject_to(self.h > 0)
            self.opti.subject_to(self.h <= self.model.dt)
            self.opti.subject_to(self.h == self.T/self.N)
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
            self.forces.append(self.opti.variable(self.model.dims + 1, self.model.n_contact))
            self.gammas.append(self.opti.variable(1))

    def __setConstraints__(self):
        self.opti.subject_to(self.states[0] == [np.pi/2, -np.pi/1.5, 0])
        self.opti.subject_to(self.states[-1][-1] == np.pi/2)
        self.opti.subject_to(self.dstates[0] == [0]*3)
        self.opti.subject_to(self.dstates[-1] == [0]*3)

        # self.phis = ca.MX.zeros(2, self.N)
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

            ## finger
            self.opti.subject_to(q_1[0:2] - q_2[0:2] + self.h * dq_2[0:2] == 0)
            self.opti.subject_to(f_2['H'] @ (dq_2[0:2] - dq_1[0:2]) -
                                 self.h * (f_2['C'] @ dq_2[0:2] + f_2['G'] - f_2['B'] @ u_2 - f_2['J_ee'].T @ f_2['lam_w']) == 0)
            ## free circle
            self.opti.subject_to(q_1[-1] - q_2[-1] + self.h * dq_2[-1] == 0)
            self.opti.subject_to(dq_1[-1] - dq_2[-1] + self.h * f_2['ddq_circle'] == 0)

            # self.phis[:, k] = kine_1['x'][:, 1] - kine_1['x'][:, 2]

            # friction constraints

            # self.opti.subject_to( (kine_1['x'][0, 1] - self.model.free_ellipse['center'][0])**2
            #                       + (kine_1['x'][1, 1] - self.model.free_ellipse['center'][1])**2
            #                       >= self.model.free_ellipse['radius']**2)

            # self.opti.subject_to(lam_1 >= 0)
            # self.opti.subject_to(f_1['phi'] >= 0)
            # self.opti.subject_to(f_1['phi'].T @ lam_1 == 0)

            lam_1_z, lam_1_xp, lam_1_xm = self.forces[k][2, 0], self.forces[k][0, 0], self.forces[k][1, 0]
            gam_1 = self.gammas[k]
            psi_1 = f_1['psi']
            self.opti.subject_to(f_1['phi'] >= 0)
            self.opti.subject_to([lam_1_z >= 0, lam_1_xp >= 0, lam_1_xm >= 0, gam_1 >= 0])
            self.opti.subject_to(self.model.mu * lam_1_z - lam_1_xp - lam_1_xm >= 0)
            self.opti.subject_to(gam_1 + psi_1 >= 0)
            self.opti.subject_to(gam_1 - psi_1 >= 0)

            self.opti.subject_to(f_1['phi'].T @ lam_1_z <= self.epsilon)
            self.opti.subject_to(f_1['phi'].T @ lam_1_z >= -self.epsilon)

            self.opti.subject_to((self.model.mu * lam_1_z - lam_1_xp - lam_1_xm).T @ lam_1_z <= self.epsilon)
            self.opti.subject_to((self.model.mu * lam_1_z - lam_1_xp - lam_1_xm).T @ lam_1_z >= -self.epsilon)

            self.opti.subject_to((gam_1 + psi_1).T @ lam_1_xp <= self.epsilon)
            self.opti.subject_to((gam_1 + psi_1).T @ lam_1_xp >= -self.epsilon)

            self.opti.subject_to((gam_1 - psi_1).T @ lam_1_xm <= self.epsilon)
            self.opti.subject_to((gam_1 - psi_1).T @ lam_1_xm >= -self.epsilon)

            # bounds model specific
            # self.opti.bounded([-np.pi]*3, q_1, [np.pi]*3)
            # self.opti.subject_to(q_1[0] <= np.pi/2)
            # self.opti.subject_to(q_1[0] >= -np.pi/2)
            # self.opti.subject_to(q_1[0] + q_1[1] <= np.pi/2)
            # self.opti.subject_to(q_1[0] + q_1[1] >= -np.pi/2)
            # self.opti.subject_to(q_1[0] <= np.pi)
            # self.opti.subject_to(q_1[0] >= -np.pi)
            # self.opti.subject_to(ca.fabs(q_1[0] + q_1[1]) > 0)

    def __setCosts__(self):
        Q = ca.diag(ca.MX([1, 1, 1]))
        R = ca.diag(ca.MX([0.1, 0.1]))
        cost = 0
        for k in range(self.N):
            q, dq = self.states[k], self.dstates[k]
            u = self.actions[k]
            cost += dq.T @ Q @ dq + u.T @ R @ u

        self.opti.minimize(cost)

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
        self.x1 = []; self.dx1 = []
        self.x2 = []; self.dx2 = []
        self.x3 = []; self.dx3 = []
        self.u1 = []; self.u2 = []

        if self.debug:
            if self.var_dt:
                self.dt = self.opti.debug.value(self.h)
            else:
                self.dt = self.h

            for i in range(self.N):
                self.x1.append(self.opti.debug.value(self.states[i][0]))
                self.x2.append(self.opti.debug.value(self.states[i][1]))
                self.x3.append(self.opti.debug.value(self.states[i][2]))

                self.dx1.append(self.solution.value(self.dstates[i][0]))
                self.dx2.append(self.solution.value(self.dstates[i][1]))
                self.dx3.append(self.solution.value(self.dstates[i][2]))

                self.u1.append(self.opti.debug.value(self.actions[i][0]))
                self.u2.append(self.opti.debug.value(self.actions[i][1]))
        else:
            if self.var_dt:
                self.dt = self.solution.value(self.h)
            else:
                self.dt = self.h

            for i in range(self.N):
                self.x1.append(self.solution.value(self.states[i][0]))
                self.x2.append(self.solution.value(self.states[i][1]))
                self.x3.append(self.solution.value(self.states[i][2]))

                self.dx1.append(self.solution.value(self.dstates[i][0]))
                self.dx2.append(self.solution.value(self.dstates[i][1]))
                self.dx3.append(self.solution.value(self.dstates[i][2]))

                self.u1.append(self.solution.value(self.actions[i][0]))
                self.u2.append(self.solution.value(self.actions[i][1]))

        self.k = self.dt/4
        self.t = np.linspace(0, self.T, int(self.T / self.k))

        self.t_points = np.linspace(0, self.N * self.dt, self.N)

        x1_spline_function = ca.interpolant('LUT', 'bspline', [self.t_points], self.x1)
        self.x1_spline = x1_spline_function(self.t)

        x2_spline_function = ca.interpolant('LUT', 'bspline', [self.t_points], self.x2)
        self.x2_spline = x2_spline_function(self.t)

        x3_spline_function = ca.interpolant('LUT', 'bspline', [self.t_points], self.x3)
        self.x3_spline = x3_spline_function(self.t)

        dx1_spline_function = ca.interpolant('LUT', 'bspline', [self.t_points], self.dx1)
        self.dx1_spline = dx1_spline_function(self.t)

        dx2_spline_function = ca.interpolant('LUT', 'bspline', [self.t_points], self.dx2)
        self.dx2_spline = dx2_spline_function(self.t)

        dx3_spline_function = ca.interpolant('LUT', 'bspline', [self.t_points], self.dx3)
        self.dx3_spline = dx3_spline_function(self.t)

        self.__plot__()

    def __plot__(self):
        fig = plt.figure()
        fig.tight_layout()

        ax1 = fig.add_subplot(311)
        ax1.grid()

        ax1.plot(self.t_points, self.x1, 'o', label='q1')
        ax1.plot(self.t_points, self.x2, 'o', label='q2')
        ax1.plot(self.t_points, self.x3, 'o', label='q3')

        ax1.plot(self.t, self.x1_spline, '-', color='black')
        ax1.plot(self.t, self.x2_spline, '-', color='black')
        ax1.plot(self.t, self.x3_spline, '-', color='black')

        ax2 = fig.add_subplot(312)
        ax2.grid()

        ax2.plot(self.t_points, self.dx1, 'o', label='dq1')
        ax2.plot(self.t_points, self.dx2, 'o', label='dq2')
        ax2.plot(self.t_points, self.dx3, 'o', label='dq3')

        ax2.plot(self.t, self.dx1_spline, '-', color='black')
        ax2.plot(self.t, self.dx2_spline, '-', color='black')
        ax2.plot(self.t, self.dx3_spline, '-', color='black')

        ax2.legend()

        ax3 = fig.add_subplot(313)
        ax3.grid()
        ax3.plot(self.t_points, self.u1, '-', label='u1')
        ax3.plot(self.t_points, self.u2, '-', label='u2')
        ax3.legend()

        plt.show()

        self.model.visualize(self.x1_spline.full(), self.x2_spline.full(),
                             self.x3_spline.full(), self.t, self.k)


Problem = NLP()
Problem.__solve__()
Problem.__interpolate__()
