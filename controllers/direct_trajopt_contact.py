""" Trajectory Optimization formulation as given in
- A Direct Method for Trajectory Optimization of Rigid Bodies Through Contact """

import casadi as ca
import numpy as np
from contact_trajectory_optimization.envs.finger_contact import FingerContact


class NLP:
    def __init__(self):
        super().__init__()
        self.model = FingerContact()
        self.model.__setOptimizationParams__(total_duration=2)

        self.opti = ca.Opti()

    def __setVariables__(self):
        self.h = self.opti.variable('h', 2, 1)
