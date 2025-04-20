import numpy as np
import pinocchio as pin
import mujoco
from mujoco import viewer
import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation as R

# ---------------------------
# PinSolver：利用 Pinocchio 进行模型加载及动力学计算
# ---------------------------
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

# ---------------------------
# JntImpedance：关节阻抗控制器（PD控制形式）
# ---------------------------
class JntImpedance:
    def __init__(self, urdf_path: str):
        self.kd_solver = PinSolver(urdf_path)
        # 阻抗参数（刚度与阻尼），可以调试以获得合适的弹性与阻尼特性
        self.k = 200.0 * np.ones(7)    # 刚度参数
        self.B = 50.0 * np.ones(7)    # 阻尼参数
        # self.k = np.array([60.0, 110.0, 60.0, 90.0, 15.0, 60.0, 15.0])  # 各关节刚度
        # self.B = np.array([11.0, 15.0, 11.0, 8.0, 0.6, 5.0, 0.6])     # 各关节阻尼


    def compute_jnt_torque(self, q_des, v_des, q_cur, v_cur):
        # 获取惯性矩阵、科氏矩阵、重力向量
        M = self.kd_solver.get_inertia_mat(q_cur)
        C = self.kd_solver.get_coriolis_mat(q_cur, v_cur)
        g = self.kd_solver.get_gravity_mat(q_cur)
        # 此处按照原代码用 C 的最后一行加重力（可根据需要调整）
        coriolis_gravity = C[-1] + g
        # 计算期望加速度（PD控制律）
        acc_desire = self.k * (q_des - q_cur) + self.B * (v_des - v_cur)
        # 利用动力学模型计算控制力矩
        tau = np.dot(M, acc_desire) + coriolis_gravity
        return tau

# ---------------------------
# Robot：集成 Mujoco 仿真与阻抗控制
# ---------------------------
class Robot:
    ACTUATORS = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7']

    def __init__(self, control_freq=50):
        # 加载 Mujoco 模型（MJCF 文件），请根据实际情况修改路径
        self.mj_model = mujoco.MjModel.from_xml_path("../../Model/Kinova_mjmodel.xml")
        self.mj_data = mujoco.MjData(self.mj_model)
        self.viewer = viewer.launch_passive(self.mj_model, self.mj_data)

        # 计算控制周期步数
        control_timestep = 1.0 / control_freq
        model_timestep = self.mj_model.opt.timestep
        self.n_substeps = int(control_timestep / model_timestep)

        # 初始化阻抗控制器，URDF 文件路径请根据实际情况修改
        self.controller = JntImpedance(urdf_path="../../Kinova_description/urdf/Kinova_description.urdf")

    def step(self, q_des):
        # 当前关节状态
        q_cur = self.mj_data.qpos.copy()
        qdot_cur = self.mj_data.qvel.copy()
        # 期望关节速度一般设为零
        v_des = np.zeros(7)
        # 计算阻抗控制产生的关节力矩
        tau = self.controller.compute_jnt_torque(q_des, v_des, q_cur, qdot_cur)
        # 将控制信号分配给各个 actuator
        for j, actuator in enumerate(self.ACTUATORS):
            self.mj_data.actuator(actuator).ctrl = tau[j]
        # 在一个控制周期内进行多步仿真
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.mj_model, self.mj_data)

    def render(self):
        if self.viewer.is_running():
            self.viewer.sync()

    def print_end_effector_pose(self):
        # 利用 Pinocchio 计算正向运动学，输出末端位姿信息
        solver = self.controller.kd_solver
        q_cur = self.mj_data.qpos.copy()
        pin.forwardKinematics(solver.model, solver.data, q_cur)
        pin.updateFramePlacements(solver.model, solver.data)
        frame_id = solver.model.getFrameId("link_tool")
        if frame_id < 0:
            print("末端框架 'link_tool' 未找到，请检查 URDF 文件！")
            return
        placement = solver.data.oMf[frame_id]
        pos = placement.translation
        rpy = R.from_matrix(placement.rotation).as_euler('xyz', degrees=True)
        print("末端位置:", pos)
        print("末端姿态 (roll, pitch, yaw in degrees):", rpy)

# ---------------------------
# GUI 部分：利用 DearPyGui 创建滑块、输入框与 Set 按钮
# ---------------------------
def set_joint_value(sender, app_data, user_data):
    slider_tag = user_data["slider_tag"]
    input_tag = user_data["input_tag"]
    try:
        value = float(dpg.get_value(input_tag))
        dpg.set_value(slider_tag, value)
    except Exception as e:
        print("输入值有误:", e)

# ---------------------------
# 主程序入口
# ---------------------------
def main():
    robot = Robot(control_freq=50)

    # 创建 DearPyGui 上下文与视窗，注册字体（请根据你的系统调整字体路径）
    dpg.create_context()
    with dpg.font_registry():
        default_font = dpg.add_font("C:/Windows/Fonts/Arial.ttf", 20)
    dpg.create_viewport(title="Joint Angle Control (Impedance)", width=600, height=500)
    dpg.bind_font(default_font)

    # 创建 GUI 窗口，每行包含滑块、文本输入框及 Set 按钮
    with dpg.window(label="Joint Angle Controls", width=600, height=500):
        for i in range(7):
            dpg.add_text(f"Joint {i+1}")
            dpg.add_slider_float(label="", tag=f"joint_{i+1}_slider",
                                 default_value=0.0, min_value=-3.14, max_value=3.14,
                                 width=300)
            # dpg.add_same_line()
            dpg.add_input_text(label="", tag=f"joint_{i+1}_input", default_value="0.0", width=100)
            # dpg.add_same_line()
            dpg.add_button(label="Set", callback=set_joint_value,
                           user_data={"slider_tag": f"joint_{i+1}_slider", "input_tag": f"joint_{i+1}_input"})

    dpg.setup_dearpygui()
    dpg.show_viewport()

    step_count = 0
    # 主循环：同时更新 GUI 与 Mujoco 仿真
    while robot.viewer.is_running() and dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()
        # 从 GUI 中读取每个关节的期望角度
        joint_angles = []
        for i in range(7):
            angle = dpg.get_value(f"joint_{i+1}_slider")
            joint_angles.append(angle)
        q_des = np.array(joint_angles)
        # 利用阻抗控制驱动机械臂
        robot.step(q_des)
        robot.render()

        step_count += 1
        if step_count % 100 == 0:
            robot.print_end_effector_pose()

    dpg.destroy_context()

if __name__ == "__main__":
    main()
