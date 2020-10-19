import casadi as ca
import numpy as np
from matplotlib import patches as pch
from matplotlib import pyplot as plt
from matplotlib import animation

"Finger Contact environment as given in - Stochastic Complementarity for Local Control of Discontinuous Dynamics"


class FingerContact:
    def __init__(self):
        super().__init__()
        self.render = False
        self.dof = 3
        self.arm = {'mass': np.array([0.3, 0.3, 0.3]).reshape((3, 1)),
                    'length': np.array([1.2, 1.2]).reshape((2, 1)),
                    'origin': np.zeros((2, 1))}

        self.free_ellipse = {'center': np.array([0., -2.]).reshape((2, 1)),  # com of ellipse
                             'axis': np.array([2.5, 1]).reshape((2, 1))}  # minor and major axis length of ellipse

        self.gravity_vector = np.array([0, 10]).reshape((2, 1))

        self.dt = 0.05

        self.state = np.array([2 * np.pi / 3, -np.pi / 3, 0]).reshape((3, 1))

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, aspect='equal')
        self.ax.set_xlim([-4, 4])
        self.ax.set_ylim([-4, 4])
        self.ax.grid()

        self.visualize()

    def __setPhysics__(self):
        q = ca.MX.sym('q', 3, 1)
        dq = ca.MX.sym('dq', 3, 1)
        ddq = ca.MX.sym('ddq', 3, 1)

        x = ca.MX.zeros(2, 3)
        x[0, 0], x[1, 0] = self.arm['origin'][0] - self.arm['length'][0] * ca.cos(q[0]), self.arm['origin'][1] - \
                           self.arm['length'][0] * ca.sin(q[0])
        x[0, 1], x[1, 1] = x[0, 0] - self.arm['length'][1] * ca.cos(q[0] + q[1]), x[1, 0] - self.arm['length'][
            1] * ca.sin(q[0] + q[1])
        x[0, 2], x[1, 2] = self.free_ellipse['center'][0], self.free_ellipse['center'][1]

        dx = ca.jtimes(x, q, dq)
        H = ca.MX.zeros(3, 3)
        for i in range(self.dof):
            if i < self.dof - 1:
                J_l = ca.jacobian(x[:, i], q)
                J_a = ca.MX.zeros(3, self.dof)
                J_a[2, :] = ca.MX([1 if j <= i else 0 for j in range(self.dof)])
                I = ca.MX.zeros(3, 3)
                I[2, 2] = (1 / 12) * self.arm['mass'][i] * self.arm['length'][i] ** 2
                H += self.arm['mass'][i] * J_l.T @ J_l + J_a.T @ I @ J_a
            # else:

        self.kinematics = ca.Function('Kinematics', [q, dq], [x, dx], ['q', 'dq'], ['x', 'dx'])

        H = ca.MX.zeros((3, 3))

    def visualize(self):
        time_template = 'time = %.1fs'
        time_text = self.ax.text(0.05, 0.9, '', transform=self.ax.transAxes)

        anchor_arm, = self.ax.plot([], [], 'o', lw=2, color='red')
        link_1, = self.ax.plot([], [], '-', lw=5, color='red')
        link_2, = self.ax.plot([], [], '-', lw=5, color='blue')
        finger_tip, = self.ax.plot([], [], 'o', lw=2, color='blue')

        anchor_ellipse, = self.ax.plot([], [], '+', lw=2, color='black')
        ellipse = pch.Ellipse(xy=self.free_ellipse['center'], width=self.free_ellipse['axis'][0],
                              height=self.free_ellipse['axis'][1], angle=(self.state[2]) * (180 / np.pi) + 180)
        ellipse.set_facecolor([0, 1, 0])

        def init():
            link_1.set_data([], [])
            link_2.set_data([], [])
            anchor_arm.set_data([], [])
            anchor_ellipse.set_data([], [])
            finger_tip.set_data([], [])
            time_text.set_text('')
            self.ax.add_patch(ellipse)
            return link_1, link_2, anchor_arm, anchor_ellipse, finger_tip, time_text, ellipse,

        def animate(i):
            line1_x = [self.arm['origin'][0], self.arm['origin'][0] - np.cos(self.state[0]) * self.arm['length'][0]]
            line1_y = [self.arm['origin'][1], self.arm['origin'][1] - np.sin(self.state[0]) * self.arm['length'][0]]
            line2_x = [line1_x[1], line1_x[1] - np.cos(self.state[1] + self.state[0]) * self.arm['length'][1]]
            line2_y = [line1_y[1], line1_y[1] - np.sin(self.state[1] + self.state[0]) * self.arm['length'][1]]

            link_1.set_data(line1_x, line1_y)
            link_2.set_data(line2_x, line2_y)

            anchor_arm.set_data([self.arm['origin'][0], self.arm['origin'][0]],
                                [self.arm['origin'][1], self.arm['origin'][1]])

            anchor_ellipse.set_data([self.free_ellipse['center'][0], self.free_ellipse['center'][0]],
                                    [self.free_ellipse['center'][1], self.free_ellipse['center'][1]])

            finger_tip.set_data([line2_x[1], line2_x[1]],
                                [line2_y[1], line2_y[1]])

            time_text.set_text(time_template % (i * self.dt))
            ellipse.center = self.free_ellipse['center']
            self.ax.add_artist(ellipse)
            # print('yes')
            return link_1, link_2, anchor_arm, anchor_ellipse, finger_tip, time_text, ellipse,

        self.ani = animation.FuncAnimation(self.fig, animate, np.arange(0, 10),
                                           interval=25)  # np.arrange for running in loop so that (i) in animate does not cross the len of x

        # ani.save('test.mp4')
        plt.show()


model = FingerContact()