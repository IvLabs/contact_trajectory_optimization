import numpy as np
import casadi as ca

from contact_trajectory_optimization.envs.terrain import Terrain


class Hopper:
    def __init__(self):
        super().__init__()
        self.render = False
        self.n_joints = 3
        self.n_contact = 1
        self.dof = 2
        self.dims = 2

        self.name = 'hopper'

        self.length = np.array([0.75, 0.75, 0.75]).reshape(3, 1)  # Base to end effector
        self.mass = np.array([20., 1., 1.]).reshape(3, 1)  # Base to end effector

        self.origin_frame = np.zeros((2, 1))

        self.inertia = self.mass * (self.length ** 2) / 12
        self.gravity = 9.81

        self.terrain = Terrain()

        # self.__setPhysics__()
        self.__setFrameTransforms__()

    def __setFrameTransforms__(self):
        """Frame Transformations from world to body and to contact frame
        b -> base
        w -> world
        l -> base link i.e <=> b
        m -> middle link
        f -> contact link
        c -> contact
        """

        q = ca.SX.sym('q', self.n_joints, 1)
        x_base = ca.SX.sym('x_base', 2, 1)

        world_frame_pos = ca.SX([0, 0])
        world_frame_ori = ca.SX([0, 0])

        body_frame_pos = x_base
        body_frame_ori = q[0]

        b_R_w = ca.SX.zeros(2, 2)
        b_R_w[0,0], b_R_w[0,1] = ca.cos(ca.pi/2 - q[0]),-ca.sin(ca.pi/2 - q[0])
        b_R_w[1,0], b_R_w[1,1] = ca.sin(ca.pi/2 - q[0]), ca.cos(ca.pi/2 - q[0])

        w_D_b = body_frame_pos - world_frame_pos

        "Joint positions"
        b_g0 = ca.SX.zeros(2, 1)
        b_p0 = ca.SX([self.length[0]/2, 0])
        w_g0 = b_g0 + w_D_b
        w_p0 = b_R_w.T @ b_p0 + w_D_b

        b_D_m = ca.SX.zeros(2, 1)
        b_D_m[0, 0] = self.length[0]/2
        b_R_m = ca.SX.zeros(2, 2)
        b_R_m[0,0], b_R_m[0,1] = ca.cos(ca.pi/2 - q[1]),-ca.sin(ca.pi/2 - q[1])
        b_R_m[1,0], b_R_m[1,1] = ca.sin(ca.pi/2 - q[1]), ca.cos(ca.pi/2 - q[1])

        m_g1 = ca.SX.zeros(2, 1)
        m_p1 = ca.SX([self.length[0] / 2, 0])
        w_g0 = m_g1
        w_p0 = b_R_w.T @ b_p0 + w_D_b

    def __setPhysics__(self):
        # Joint State, Base to end effector
        q = ca.SX.sym('q', self.n_joints, 1)
        dq = ca.SX.sym('dq', self.n_joints, 1)
        ddq = ca.SX.sym('ddq', self.n_joints, 1)

        # Inputs and external force
        u = ca.SX.sym('u', self.dof, 1)
        lam = ca.SX.sym('lambda', 3, self.n_contact)

        # Base state
        x_base = ca.SX.sym('x_base', 2, 1)
        dx_base = ca.SX.sym('dx_base', 2, 1)
        ddx_base = ca.SX.sym('ddx_base', 2, 1)

        """Hopper Dynamics"""

        "End Effector pos"
        pe = ca.SX.zeros(3, self.n_joints)
        pe[0, 0], pe[1, 0] = x_base[0] + (self.length[0]/2) * ca.sin(q[0]), x_base[1] - (self.length[1]/2) * ca.cos(q[0])
        pe[0, 1], pe[1, 1] = pe[0, 0] + self.length[1] * ca.sin(ca.sum1(q[0:2])), pe[1, 0] - self.length[1] * ca.cos(ca.sum1(q[0:2]))
        pe[0, 2], pe[1, 2] = pe[0, 1] + self.length[2] * ca.sin(ca.sum1(q)), pe[1, 1] - self.length[2] * ca.cos(ca.sum1(q))

        # COG
        g = ca.SX.zeros(3, self.n_joints)
        g[0, 0], g[1, 0] = x_base[0], x_base[1]
        g[0, 1], g[1, 1] = pe[0, 0] + (self.length[1]/2) * ca.sin(ca.sum1(q[0:2])), pe[1, 0] - (self.length[1]/2) * ca.cos(ca.sum1(q[0:2]))
        g[0, 2], g[1, 2] = pe[0, 1] + (self.length[2]/2) * ca.sin(ca.sum1(q)), pe[1, 1] - (self.length[2]/2) * ca.cos(ca.sum1(q))

        a = ca.SX.zeros(3, self.n_joints)
        temp = ca.DM.ones(ca.Sparsity.lower(3))
        a[2, :] = (temp @ q).T

        J_ee = ca.jacobian(pe[:, 2], q)

        "For inertia matrix"
        H = ca.SX.zeros(self.n_joints, self.n_joints)

        for i in range(self.dof):
            J_l = ca.jacobian(pe[:, i], q)
            J_a = ca.jacobian(a[:, i], q)
            # print(J_l.shape)
            # print('------------')
            # print(J_a.shape)

            I = ca.SX.zeros(3, 3)
            I[2, 2] = (1 / 12) * self.mass[i] * self.length[i] ** 2  # Rod

            H += self.mass[i] * J_l.T @ J_l + J_a.T @ I @ J_a

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
        V = self.gravity * ca.sum1(self.mass * g[1, :].T)
        G = ca.jacobian(V, q).T

        "For B matrix"
        B = ca.DM([[-1, 0],
                   [1, -1],
                   [0, 1]])

        "For external force"
        pe_terrain = ca.SX.zeros(2, 1)
        pe_terrain[0, 0] = pe[0, -1]
        pe_terrain[1, 0] = self.terrain.heightMap(pe[0, -1])
        phi = pe[1, -1] - pe_terrain[1, 0]

        theta_contact = ca.acos(ca.dot(self.terrain.heightMapNormalVector(pe[0, -1]), ca.DM([0, 1])))
        # rotate lambda force represented in contact frame to world frame
        Rot_contact = ca.SX.zeros(2, 2)
        Rot_contact[0, 0], Rot_contact[0, 1] = ca.cos(theta_contact), ca.sin(theta_contact)
        Rot_contact[1, 0], Rot_contact[1, 1] = -ca.sin(theta_contact), ca.cos(theta_contact)
        lam_c = ca.SX.zeros(2, 1)
        lam_c[0, 0], lam_c[1, 0] = lam[0, 0] - lam[1, 0], lam[2, 0]
        lam_w = Rot_contact @ lam_c

        # print(H)


model = Hopper()
