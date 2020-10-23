"""Finger Contact environment as given in
- Stochastic Complementarity for Local Control of Discontinuous Dynamics"""

import casadi as ca
import numpy as np
from matplotlib import patches as pch
from matplotlib import pyplot as plt
from matplotlib import animation


class FingerContact:
    def __init__(self):
        super().__init__()
        self.render = False
        self.n_joints = 3
        self.n_contact = 1
        self.dof = 2
        self.dims = 2

        self.arm = {'mass': np.array([3, 0.3, 0.3]).reshape((3, 1)), # also mass of circle
                    'length': np.array([1.3, 1.3]).reshape((2, 1)),
                    'origin': np.zeros((2, 1))}

        self.free_circle = {'center': np.array([0., -2.]).reshape((2, 1)),  # com of ellipse
                            'radius': np.array([0.5]).reshape((1, 1))}  # minor and major axis length of ellipse

        self.gravity = -9.81
        self.mu = 1.
        self.dt = 0.1

        self.state = np.array([(180/np.pi)*45, -(180/np.pi)*3, 0]).reshape((3, 1))

        # self.visualize()
        self.__setPhysics__()

    def __setPhysics__(self):
        q = ca.SX.sym('q', self.n_joints, 1)
        dq = ca.SX.sym('dq', self.n_joints, 1)
        ddq = ca.SX.sym('ddq', self.n_joints, 1)
        lam = ca.SX.sym('lambda', 3, self.n_contact)

        "End Effector pos"
        x = ca.SX.zeros(2, self.n_joints)
        x[0, 0], x[1, 0] = self.arm['origin'][0] + self.arm['length'][0] * ca.sin(q[0]), self.arm['origin'][1] - self.arm['length'][0] * ca.cos(q[0])
        x[0, 1], x[1, 1] = x[0, 0] + self.arm['length'][1] * ca.sin(q[0] + q[1]), x[1, 0] - self.arm['length'][1] * ca.cos(q[0] + q[1])
        x[0, 2], x[1, 2] = self.free_circle['center'][0], self.free_circle['center'][1]

        a = ca.SX.zeros(1, self.n_joints)
        temp = ca.DM.ones(ca.Sparsity.diag(3)); temp[1, 0] = 1
        a[0, :] = temp @ q

        J_ee = ca.jacobian(x[:, 1], q)

        J_ee_b = ca.SX.ones(3, 2)
        J_ee_b[0, 0] = self.arm['length'][1] + self.arm['length'][0]*ca.cos(q[1])
        J_ee_b[0, 1] = self.arm['length'][1]
        J_ee_b[1, 0] = - self.arm['length'][1]*ca.sin(q[1])
        J_ee_b[1, 1] = 0

        J_ee_s = ca.SX.ones(3, 2)
        J_ee_s[0, 0] = 0
        J_ee_s[0, 1] = self.arm['length'][1]*ca.cos(q[0])
        J_ee_s[1, 0] = 0
        J_ee_s[1, 1] = - self.arm['length'][1]*ca.sin(q[1])

        dx = ca.jtimes(x, q, dq)
        da = ca.jtimes(a, q, dq)

        "For inertia matrix"
        H = ca.SX.zeros(self.n_joints, self.n_joints)
        for i in range(self.n_joints):
            J_l = ca.jacobian(x[:, i], q)
            J_a = ca.jacobian(a[:, i], q).T
            # print(J_l.shape)
            # print('------------')

            I = ca.SX.zeros(3, 3)

            if i < self.n_joints - 1:
                I[2, 2] = (1 / 12) * self.arm['mass'][i] * self.arm['length'][i] ** 2  # Rod
            else:
                I[2, 2] = (1 / 2) * self.arm['mass'][i] * np.sum(self.free_circle['radius'] ** 2)  # Ellipse

            H += self.arm['mass'][i] * J_l.T @ J_l + J_a.T @ I @ J_a

        "For coriolis + centrifugal matrix"
        C = ca.SX.zeros(self.n_joints, self.n_joints)
        for i in range(self.n_joints):
            for j in range(self.n_joints):
                sum_ = 0
                for k in range(self.n_joints):
                    c_ijk = ca.jacobian(H[i, j], q[k]) - (1/2)*ca.jacobian(H[j, k], q[i])
                    sum_ += c_ijk @ dq[j] @ dq[k]
                C[i, j] = sum_

        "For G matrix"
        V = self.gravity * ca.sum2(self.arm['mass'].T * x[1, :])
        G = ca.jacobian(V, q).T

        # For B matrix
        B = ca.DM.zeros(self.n_joints, self.dof)
        B[0, 0], B[1, 1] = 1, 1

        "For external force"
        theta_contact = ca.atan2(x[0, 1] - self.free_circle['center'][0], x[1, 1] - self.free_circle['center'][1])
        p_contact = ca.SX.zeros(2, 1)
        p_contact[0, 0], p_contact[1, 0] = self.free_circle['radius']*ca.cos(theta_contact), self.free_circle['radius']*ca.sin(theta_contact)
        phi = x[:, 1] - p_contact

        # rotate lambda force represented in contact frame to world frame
        Rot_contact = ca.SX.zeros(2, 2)
        Rot_contact[0, 0], Rot_contact[0, 1] = ca.cos(theta_contact), ca.sin(theta_contact)
        Rot_contact[1, 0], Rot_contact[1, 1] = - ca.sin(theta_contact), ca.cos(theta_contact)
        lam_c = ca.SX.zeros(2, 1)
        lam_c[0, 0], lam_c[1, 0] = lam[0, 0] - lam[1, 0], lam[2, 0]
        lam_w = Rot_contact @ lam_c

        # For tangential vel @ contact
        normal_vec_contact = Rot_contact.T @ ca.SX.ones(2, 1)
        tangent_vec_contact = ca.DM([[0, -1], [1, 0]]) @ normal_vec_contact

        ee_vel_b = J_ee_b @ dq[0:2]
        Rot_q1q2 = ca.SX.zeros(2, 2)
        Rot_q1q2[0, 0], Rot_q1q2[0, 1] = ca.cos(q[0] + q[1]), -ca.sin(q[0] + q[1])
        Rot_q1q2[1, 0], Rot_q1q2[1, 1] = ca.sin(q[0] + q[1]),  ca.cos(q[0] + q[1])
        ee_vel_s = Rot_q1q2 @ ee_vel_b[0:2]

        self.dynamics = ca.Function('Dynamics', [q, dq, lam], [H, C, G, B, phi, J_ee, J_ee_b, J_ee_s, lam_c, lam_w],
                        ['q', 'dq', 'lam'], ['H', 'C', 'G', 'B', 'phi', 'J_ee', 'J_ee_b', 'J_ee_s', 'lam_c', 'lam_w'])
        self.kinematics = ca.Function('Kinematics', [q, dq], [x, dx, a, da],
                                    ['q', 'dq'], ['x', 'dx', 'a', 'da'])

    def visualize(self, x1, x2, x3, t):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, aspect='equal')
        self.ax.set_xlim([-4, 4])
        self.ax.set_ylim([-4, 4])
        self.ax.grid()

        time_template = 'time = %.1fs'
        time_text = self.ax.text(0.05, 0.9, '', transform=self.ax.transAxes)

        anchor_arm, = self.ax.plot([], [], 'o', lw=2, color='red')
        link_1, = self.ax.plot([], [], '-', lw=5, color='red')
        link_2, = self.ax.plot([], [], '-', lw=5, color='blue')
        finger_tip, = self.ax.plot([], [], 'o', lw=2, color='blue')

        anchor_circle, = self.ax.plot([], [], '+', lw=1, color='black')
        circle = pch.Circle(xy=self.free_circle['center'], radius=self.free_circle['radius'])
        circle.set_facecolor([0, 1, 0])

        def init():
            link_1.set_data([], [])
            link_2.set_data([], [])
            anchor_arm.set_data([], [])
            anchor_circle.set_data([], [])
            finger_tip.set_data([], [])
            time_text.set_text('')
            self.ax.add_patch(circle)
            return link_1, link_2, anchor_arm, anchor_circle, finger_tip, time_text, circle,

        def animate(i):
            line1_x = [self.arm['origin'][0], self.arm['origin'][0] - np.cos(x1[i]) * self.arm['length'][0]]
            line1_y = [self.arm['origin'][1], self.arm['origin'][1] - np.sin(x1[i]) * self.arm['length'][0]]
            line2_x = [line1_x[1], line1_x[1] - np.cos(x1[i] + x2[i]) * self.arm['length'][1]]
            line2_y = [line1_y[1], line1_y[1] - np.sin(x1[i] + x2[i]) * self.arm['length'][1]]

            link_1.set_data(line1_x, line1_y)
            link_2.set_data(line2_x, line2_y)

            anchor_arm.set_data([self.arm['origin'][0], self.arm['origin'][0]],
                                [self.arm['origin'][1], self.arm['origin'][1]])

            anchor_circle.set_data([self.free_circle['center'][0], self.free_circle['center'][0] + np.cos(x3[i])*self.free_circle['radius']],
                                   [self.free_circle['center'][1], self.free_circle['center'][1] + np.sin(x3[i])*self.free_circle['radius']])

            finger_tip.set_data([line2_x[1], line2_x[1]],
                                [line2_y[1], line2_y[1]])

            time_text.set_text(time_template % (i * 0.05))
            self.ax.add_artist(circle)
            # print('yes')
            return link_1, link_2, anchor_arm, anchor_circle, finger_tip, time_text, circle,

        self.ani = animation.FuncAnimation(self.fig, animate, np.arange(0, len(t)),
                                           interval=25)  # np.arrange for running in loop so that (i) in animate does not cross the len of x

        # ani.save('test.mp4')
        plt.show()


model = FingerContact()
