import numpy as np

from pinocchio_dynamic_solver import PinSolver

class JntImpedance:
    def __init__(
            self,
            urdf_path: str,
    ):
        self.kd_solver = PinSolver(urdf_path)

        # hyperparameters of impedance controller
        self.k = 6.0 * np.ones(7)
        self.B = 0.8 * np.ones(7)

    def compute_jnt_torque(self, q_des, v_des, q_cur, v_cur):
        """ robot的关节空间控制的计算公式
            Compute desired torque with robot dynamics modeling:
            > M(q)qdd + C(q, qd)qd + G(q) + tau_F(qd) = tau_ctrl + tau_env

        :param q_des: desired joint position
        :param v_des: desired joint velocity
        :param q_cur: current joint position
        :param v_cur: current joint velocity
        :return: desired joint torque
        """
        M = self.kd_solver.get_inertia_mat(q_cur)
        C = self.kd_solver.get_coriolis_mat(q_cur, v_cur)
        g = self.kd_solver.get_gravity_mat(q_cur)
        coriolis_gravity = C[-1] + g

        acc_desire = self.k * (q_des - q_cur) + self.B * (v_des - v_cur)
        tau = np.dot(M, acc_desire) + coriolis_gravity
        return tau
