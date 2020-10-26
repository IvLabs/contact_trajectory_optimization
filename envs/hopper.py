import numpy as np
import casadi as ca


class Hopper:
    def __init__(self):
        super().__init__()
        self.render = False
        self.n_joints = 2
        self.n_contact = 1
        self.dof = 2
        self.dims = 2

        self.name = 'hopper'

        self.length = np.array([0.75, 0.75, 0.75]).reshape(3, 1)  # Base to end effector
        self.mass = np.array([20., 1., 1.]).reshape(3, 1)  # Base to end effector

        self.origin_frame = np.zeros((2, 1))

        self.inertia = self.mass * (self.length**2)/12
        self.gravity = 9.81

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
        pe = ca.SX.zeros(2, self.n_joints + 1)
        pe[0, 0], pe[1, 0] = x_base[0] + self.length[0] * ca.sin(q[0]), x_base[1] - self.length[1] * ca.cos(q[0])
        pe[0, 1], pe[1, 1] = pe[0, 0] + self.length[1] * ca.sin(ca.sum1(q[0:2])), pe[1, 0] - self.length[1] * ca.cos(q[0] + q[1])
        pe[0, 1], pe[1, 1] = pe[0, 0] + self.length[1] * ca.sin(q[0] + q[1]), pe[1, 0] - self.length[1] * ca.cos(q[0] + q[1])

        a = ca.SX.zeros(3, self.dof)
        temp = ca.DM.ones(ca.Sparsity.diag(2)); temp[1, 0] = 1
        a[2, :] = temp @ q

