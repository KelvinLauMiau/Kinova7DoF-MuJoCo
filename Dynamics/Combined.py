import mujoco
from mujoco import viewer
import numpy as np
import pinocchio as pin

# ------------------------------
# PinSolver (来自 pinocchio_dynamic_solver.py)
# ------------------------------
class PinSolver:
    """ Pinocchio solver for kinematics and dynamics """
    def __init__(self, urdf_path: str):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self._JOINT_NUM = self.model.nq

    def get_inertia_mat(self, q):
        """ Computing the inertia matrix in the joint frame """
        return pin.crba(self.model, self.data, q).copy()

    def get_coriolis_mat(self, q, qdot):
        """ Computing the Coriolis matrix in the joint frame """
        return pin.computeCoriolisMatrix(self.model, self.data, q, qdot).copy()

    def get_gravity_mat(self, q):
        """ Computing the gravity matrix in the joint frame """
        return pin.computeGeneralizedGravity(self.model, self.data, q).copy()


# ------------------------------
# JntImpedance (来自 joint_impedance_controller.py)
# ------------------------------
class JntImpedance:
    def __init__(self, urdf_path: str):
        self.kd_solver = PinSolver(urdf_path)
        # Hyperparameters for impedance controller: low stiffness and damping for dragging teaching
        self.k = 6.0 * np.ones(7)
        self.B = 0.8 * np.ones(7)

    def compute_jnt_torque(self, q_des, v_des, q_cur, v_cur):
        """
        Compute desired joint torque using dynamics:
          M(q)qdd + C(q, qd)qd + G(q) + tau_F(qd) = tau_ctrl + tau_env

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


# ------------------------------
# 主程序 (来自 dynamics_dragteaching_demo.py)
# ------------------------------
class Robot:
    ACTUATORS = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7']

    def __init__(self, control_freq=20) -> None:
        self.mj_model = mujoco.MjModel.from_xml_path(filename='../Model/ActualArm/Kinova_mjmodel.xml')
        self.mj_data = mujoco.MjData(self.mj_model)
        self.viewer = viewer.launch_passive(self.mj_model, self.mj_data)

        # 计算控制频率
        control_timestep = 1.0 / control_freq
        model_timestep = self.mj_model.opt.timestep
        self._n_substeps = int(control_timestep / model_timestep)

        # 初始化控制器 (拖动示教)
        self.controller = JntImpedance(urdf_path='../Kinova_description/urdf/Kinova_description.urdf')

    def render(self):
        if self.viewer.is_running():
            self.viewer.sync()

    def step(self, action: np.ndarray):
        for i in range(self._n_substeps):
            self.inner_step(action)
            mujoco.mj_step(self.mj_model, self.mj_data)

    def inner_step(self, action):
        torque = self.controller.compute_jnt_torque(
            q_des=action,
            v_des=np.zeros(7),
            q_cur=self.mj_data.qpos,
            v_cur=self.mj_data.qvel,
        )
        for j, per_actuator_index in enumerate(self.ACTUATORS):
            self.mj_data.actuator(per_actuator_index).ctrl = torque[j]


if __name__ == '__main__':
    robot = Robot()
    for _ in range(int(1e5)):
        # 此处 action 使用当前 qpos，达到拖动示教效果
        robot.step(robot.mj_data.qpos)
        robot.render()
