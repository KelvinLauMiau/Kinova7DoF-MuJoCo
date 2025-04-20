import numpy as np
import pinocchio as pin
import mujoco
from mujoco import viewer
from scipy.spatial.transform import Rotation as R

# ----------------------------------------
# 定义 Pinocchio 求解器，用于动力学与正向运动学计算
# ----------------------------------------
class PinSolver:
    def __init__(self, urdf_path: str):
        # 根据 URDF 构建机器人模型
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

    def get_inertia_mat(self, q):
        return pin.crba(self.model, self.data, q).copy()

    def get_coriolis_mat(self, q, qdot):
        return pin.computeCoriolisMatrix(self.model, self.data, q, qdot).copy()

    def get_gravity_mat(self, q):
        return pin.computeGeneralizedGravity(self.model, self.data, q).copy()

# ----------------------------------------
# 定义关节阻抗控制器
# ----------------------------------------
class JntImpedance:
    def __init__(self, urdf_path: str):
        self.kd_solver = PinSolver(urdf_path)
        self.k = 6.0 * np.ones(7)
        self.B = 0.8 * np.ones(7)

    def compute_jnt_torque(self, q_des, v_des, q_cur, v_cur):
        M = self.kd_solver.get_inertia_mat(q_cur)
        C = self.kd_solver.get_coriolis_mat(q_cur, v_cur)
        g = self.kd_solver.get_gravity_mat(q_cur)
        # 这里采用原代码中的写法：C[-1] + g
        coriolis_gravity = C[-1] + g
        acc_desire = self.k * (q_des - q_cur) + self.B * (v_des - v_cur)
        tau = np.dot(M, acc_desire) + coriolis_gravity
        return tau

# ----------------------------------------
# 定义 Drag Teaching Forward_Demo 的机器人类
# ----------------------------------------
class Robot:
    ACTUATORS = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7']

    def __init__(self, control_freq=20):
        # 加载 Mujoco 模型（MJCF 文件）
        # 请根据实际情况调整路径（例如："Kinova_mjmodel.xml"）
        self.mj_model = mujoco.MjModel.from_xml_path("../../Model/Kinova_mjmodel.xml")
        self.mj_data = mujoco.MjData(self.mj_model)
        self.viewer = viewer.launch_passive(self.mj_model, self.mj_data)

        # 计算控制频率
        control_timestep = 1.0 / control_freq
        model_timestep = self.mj_model.opt.timestep
        self._n_substeps = int(control_timestep / model_timestep)

        # 初始化阻抗控制器，URDF 文件路径需根据实际情况修改
        self.controller = JntImpedance(urdf_path="../../Kinova_description/urdf/Kinova_description.urdf")

    def step(self, action: np.ndarray):
        """在每个控制周期内进行多次模型步进"""
        for i in range(self._n_substeps):
            self.inner_step(action)
            mujoco.mj_step(self.mj_model, self.mj_data)

    def inner_step(self, action):
        # 使用阻抗控制计算所需关节力矩
        torque = self.controller.compute_jnt_torque(
            q_des=action,
            v_des=np.zeros(7),
            q_cur=self.mj_data.qpos,
            v_cur=self.mj_data.qvel,
        )
        # 将力矩值写入各个 actuator
        for j, actuator_name in enumerate(self.ACTUATORS):
            self.mj_data.actuator(actuator_name).ctrl = torque[j]

    def render(self):
        """渲染 Mujoco 仿真窗口"""
        if self.viewer.is_running():
            self.viewer.sync()

    def print_end_effector_pose(self):
        """
        利用 Pinocchio 计算正向运动学，
        并打印末端框架（假定名称为 "link_tool"）的位置与姿态（roll, pitch, yaw，角度制）
        """
        solver = self.controller.kd_solver
        # 取当前仿真中各关节角度（假定与 Pinocchio 模型一致）
        q = self.mj_data.qpos.copy()
        pin.forwardKinematics(solver.model, solver.data, q)
        pin.updateFramePlacements(solver.model, solver.data)
        frame_id = solver.model.getFrameId("link_tool")
        if frame_id < 0:
            print("末端框架 'link_tool' 未找到，请检查 URDF 文件中的框架名称。")
            return
        placement = solver.data.oMf[frame_id]
        pos = placement.translation
        rot = placement.rotation
        # 将旋转矩阵转换为 Euler 角 (roll, pitch, yaw)
        rpy = R.from_matrix(rot).as_euler('xyz', degrees=True)
        print("末端位置:", pos)
        print("末端姿态 (旋转矩阵):", rot)
        print("末端姿态 (roll, pitch, yaw in degrees):", rpy)

# ----------------------------------------
# 主程序入口
# ----------------------------------------
if __name__ == "__main__":
    robot = Robot(control_freq=20)
    num_steps = int(1e5)
    for step in range(num_steps):
        # drag teaching demo 中采用零力拖动，即期望关节位置等于当前状态
        robot.step(robot.mj_data.qpos)
        robot.render()
        # 为避免过多输出，每 100 步打印一次末端位姿
        if step % 200 == 0:
            robot.print_end_effector_pose()
