import mujoco
from mujoco import viewer
import numpy as np
import pinocchio as pin

# -------------------------------
# 内嵌 PinSolver：利用 Pinocchio 加载 URDF 并计算动力学量
# -------------------------------
class PinSolver:
    def __init__(self, urdf_path: str):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

    def get_inertia_mat(self, q):
        return pin.crba(self.model, self.data, q).copy()

    def get_coriolis_mat(self, q, qdot):
        return pin.computeCoriolisMatrix(self.model, self.data, q, qdot).copy()

    def get_gravity_mat(self, q):
        return pin.computeGeneralizedGravity(self.model, self.data, q).copy()

# -------------------------------
# 内嵌 JntImpedance：阻抗控制器（拖动示教）——采用低刚度与低阻尼
# -------------------------------
class JntImpedance:
    def __init__(self, urdf_path: str):
        self.kd_solver = PinSolver(urdf_path)
        self.k = 6.0 * np.ones(7)
        self.B = 0.8 * np.ones(7)

    def compute_jnt_torque(self, q_des, v_des, q_cur, v_cur):
        M = self.kd_solver.get_inertia_mat(q_cur)
        C = self.kd_solver.get_coriolis_mat(q_cur, v_cur)
        g = self.kd_solver.get_gravity_mat(q_cur)
        coriolis_gravity = C[-1] + g
        acc_desire = self.k * (q_des - q_cur) + self.B * (v_des - v_cur)
        tau = np.dot(M, acc_desire) + coriolis_gravity
        return tau

# -------------------------------
# TargetArmRobot：仅针对 Target Arm 的拖动示教
# -------------------------------
class TargetArmRobot:
    # 假设 Target Arm 的 actuator 名称为 target_a1 ~ target_a7，对应 mj_data.ctrl 的索引 7~14
    ACTUATORS = ['target_a1', 'target_a2', 'target_a3', 'target_a4', 'target_a5', 'target_a6', 'target_a7']

    def __init__(self, control_freq=20):
        # 请确保你的 MJCF 文件中 Target Arm 部分配置正确
        self.mj_model = mujoco.MjModel.from_xml_path(filename='../../Model/Kinova_System.xml')
        self.mj_data = mujoco.MjData(self.mj_model)
        self.viewer = viewer.launch_passive(self.mj_model, self.mj_data)

        # 计算控制周期内的步数
        control_timestep = 1.0 / control_freq
        model_timestep = self.mj_model.opt.timestep
        self.n_substeps = int(control_timestep / model_timestep)

        # 初始化拖动示教控制器，采用低刚度与低阻尼
        self.controller = JntImpedance(urdf_path='../../Kinova_description/urdf/Kinova_description.urdf')

        # 初始目标角度设为当前 Target Arm 状态（假设索引 7~14）
        self.q_des = self.mj_data.qpos[7:14].copy()

    def step(self):
        # 读取 Target Arm 当前状态（索引 7~14）
        q_cur = self.mj_data.qpos[7:14].copy()
        qdot_cur = self.mj_data.qvel[7:14].copy()
        v_des = np.zeros(7)
        # 计算控制力矩
        tau = self.controller.compute_jnt_torque(q_des=self.q_des,
                                                   v_des=v_des,
                                                   q_cur=q_cur,
                                                   v_cur=qdot_cur)
        # 将控制信号赋给 Target Arm actuators（ctrl[7:14]）
        self.mj_data.ctrl[7:14] = tau
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.mj_model, self.mj_data)

    def render(self):
        if self.viewer.is_running():
            self.viewer.sync()

if __name__ == '__main__':
    robot = TargetArmRobot(control_freq=20)
    while robot.viewer.is_running():
        robot.step()
        robot.render()
