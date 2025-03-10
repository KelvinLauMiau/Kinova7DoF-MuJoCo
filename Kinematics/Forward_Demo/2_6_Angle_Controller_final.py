import numpy as np
import pinocchio as pin
import mujoco
from mujoco import viewer
import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation as R

# --------------------------------------------------
# PinSolver：利用 Pinocchio 加载模型并计算相关动力学量
# --------------------------------------------------
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

# --------------------------------------------------
# JntImpedance：关节阻抗控制器（PD控制形式）
# --------------------------------------------------
class JntImpedance:
    def __init__(self, urdf_path: str):
        self.kd_solver = PinSolver(urdf_path)
        # 为所有关节统一设置刚度与阻尼（后续可以改为单独设置）
        self.k = 200.0 * np.ones(7)    # 刚度参数
        self.B = 50.0 * np.ones(7)     # 阻尼参数

    def compute_jnt_torque(self, q_des, v_des, q_cur, v_cur):
        # 获取机器人动力学量
        M = self.kd_solver.get_inertia_mat(q_cur)
        C = self.kd_solver.get_coriolis_mat(q_cur, v_cur)
        g = self.kd_solver.get_gravity_mat(q_cur)
        # 此处使用 C 的最后一行加重力作为补偿（可根据需要修改）
        coriolis_gravity = C[-1] + g
        # PD 控制律：根据目标与当前状态计算期望加速度
        acc_desire = self.k * (q_des - q_cur) + self.B * (v_des - v_cur)
        tau = np.dot(M, acc_desire) + coriolis_gravity
        return tau

# --------------------------------------------------
# Robot：集成 Mujoco 仿真与阻抗控制
# --------------------------------------------------
class Robot:
    ACTUATORS = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7']

    def __init__(self, control_freq=50, mjcf_path="../../Model/Kinova_mjmodel.xml",
                 urdf_path="../../Kinova_description/urdf/Kinova_description.urdf"):
        # 加载 Mujoco 模型
        self.mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.viewer = viewer.launch_passive(self.mj_model, self.mj_data)

        # 根据控制频率计算每个控制周期内的步数
        self.control_timestep = 1.0 / control_freq
        model_timestep = self.mj_model.opt.timestep
        self.n_substeps = int(self.control_timestep / model_timestep)

        # 初始化阻抗控制器（PD形式）
        self.controller = JntImpedance(urdf_path)

    def step(self, q_des):
        q_cur = self.mj_data.qpos.copy()
        qdot_cur = self.mj_data.qvel.copy()
        v_des = np.zeros(7)  # 期望速度设为零
        tau = self.controller.compute_jnt_torque(q_des, v_des, q_cur, qdot_cur)
        # 将计算得到的力矩分配到各个 actuator
        for j, actuator in enumerate(self.ACTUATORS):
            self.mj_data.actuator(actuator).ctrl = tau[j]
        # 在一个控制周期内进行多步仿真更新
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.mj_model, self.mj_data)

    def render(self):
        if self.viewer.is_running():
            self.viewer.sync()

    def print_end_effector_pose(self):
        # 利用 Pinocchio 计算正向运动学，输出末端位姿
        solver = self.controller.kd_solver
        q_cur = self.mj_data.qpos.copy()
        pin.forwardKinematics(solver.model, solver.data, q_cur)
        pin.updateFramePlacements(solver.model, solver.data)
        frame_id = solver.model.getFrameId("link_tool")
        if frame_id < 0:
            print("末端框架 'link_tool' 未找到！")
            return
        placement = solver.data.oMf[frame_id]
        pos = placement.translation
        rot = placement.rotation
        rpy = R.from_matrix(rot).as_euler('xyz', degrees=True)
        print("末端位置:", pos)
        print("末端姿态 (旋转矩阵):")
        print(rot)
        print("末端姿态 (roll, pitch, yaw in degrees):", rpy)

# --------------------------------------------------
# GUI 部分：使用 DearPyGui 实现关节角度控制界面
# --------------------------------------------------
def set_joint_value(sender, app_data, user_data):
    slider_tag = user_data["slider_tag"]
    input_tag = user_data["input_tag"]
    try:
        value = float(dpg.get_value(input_tag))
        dpg.set_value(slider_tag, value)
    except Exception as e:
        print("输入值有误:", e)

def create_gui():
    # 创建 GUI 上下文，并注册字体（请根据实际系统修改字体路径）
    dpg.create_context()
    with dpg.font_registry():
        default_font = dpg.add_font("C:/Windows/Fonts/Arial.ttf", 48)
    dpg.create_viewport(title="Joint Angle Control (Impedance)", width=800, height=550)
    dpg.bind_font(default_font)
    # 创建控制窗口：每行显示一个关节的滑块、文本输入框及 Set 按钮
    with dpg.window(label="Joint Angle Controls", width=775, height=550):
        for i in range(7):
            with dpg.group(horizontal=True):
                dpg.add_text(f"Joint {i+1}")
                dpg.add_slider_float(label="", tag=f"joint_{i+1}_slider",
                                     default_value=0.0, min_value=-3.14, max_value=3.14,
                                     width=300)
                dpg.add_input_text(label="", tag=f"joint_{i+1}_input", default_value="0.0", width=150)
                dpg.add_button(label="  SET  ", callback=set_joint_value,
                               user_data={"slider_tag": f"joint_{i+1}_slider", "input_tag": f"joint_{i+1}_input"})
    dpg.setup_dearpygui()
    dpg.show_viewport()

# --------------------------------------------------
# 主函数：整合 Robot 与 GUI
# --------------------------------------------------
def main():
    # 实例化机器人，传入相应模型的路径
    robot = Robot(control_freq=50,
                  mjcf_path="../../Model/ActualArm/Kinova_mjmodel.xml",
                  urdf_path="../../Kinova_description/urdf/Kinova_description.urdf")
    create_gui()
    step_count = 0
    while robot.viewer.is_running() and dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()
        # 从 GUI 读取每个关节的目标角度
        joint_angles = []
        for i in range(7):
            angle = dpg.get_value(f"joint_{i+1}_slider")
            joint_angles.append(angle)
        q_des = np.array(joint_angles)
        # 使用阻抗控制驱动机器人
        robot.step(q_des)
        robot.render()
        step_count += 1
        if step_count % 100 == 0:
            robot.print_end_effector_pose()
    dpg.destroy_context()

if __name__ == "__main__":
    main()
