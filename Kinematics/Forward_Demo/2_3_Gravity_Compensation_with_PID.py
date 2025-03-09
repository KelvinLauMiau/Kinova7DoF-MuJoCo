import numpy as np
import mujoco
from mujoco import viewer
import dearpygui.dearpygui as dpg
import pinocchio as pin
from scipy.spatial.transform import Rotation as R

# ---------------------------
# 定义 PID 控制器类
# ---------------------------
class PIDController:
    def __init__(self, Kp, Ki, Kd, n_joints):
        self.Kp = np.array(Kp)
        self.Ki = np.array(Ki)
        self.Kd = np.array(Kd)
        self.integral = np.zeros(n_joints)

    def reset(self):
        self.integral = np.zeros_like(self.integral)

    def update(self, q_target, q, qdot, dt):
        error = q_target - q
        self.integral += error * dt
        # 注意：这里用 -qdot 表示负反馈（阻尼项）
        tau = self.Kp * error + self.Ki * self.integral - self.Kd * qdot
        return tau

# ---------------------------
# 定义机器人仿真类（含 PID 控制及重力补偿）
# ---------------------------
class Robot:
    ACTUATORS = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7']

    def __init__(self, control_freq=50):
        # 加载 Mujoco 模型（请根据实际情况修改 MJCF 文件路径）
        self.mj_model = mujoco.MjModel.from_xml_path("../../Model/Kinova_mjmodel.xml")
        self.mj_data = mujoco.MjData(self.mj_model)
        self.viewer = viewer.launch_passive(self.mj_model, self.mj_data)

        self.control_freq = control_freq
        self.control_timestep = 1.0 / control_freq
        model_timestep = self.mj_model.opt.timestep
        self.n_substeps = int(self.control_timestep / model_timestep)

        # 设置 PID 增益（需要根据实际调试）
        Kp = [0.1] * 7
        Ki = [0.0] * 7
        Kd = [0.01] * 7
        self.pid = PIDController(Kp, Ki, Kd, 7)

        # 加载 Pinocchio 模型，用于计算重力补偿和末端正向运动学（请根据实际情况修改 URDF 路径）
        urdf_path = "../../Kinova_description/urdf/Kinova_description.urdf"
        self.pin_model = pin.buildModelFromUrdf(urdf_path)
        self.pin_data = self.pin_model.createData()

    def step(self, q_target):
        # 当前关节状态
        q_cur = self.mj_data.qpos.copy()
        qdot_cur = self.mj_data.qvel.copy()

        # 计算 PID 输出（单位为关节力矩）
        tau_pid = self.pid.update(q_target, q_cur, qdot_cur, self.control_timestep)

        # 重力补偿（利用 Pinocchio 计算）
        g = pin.computeGeneralizedGravity(self.pin_model, self.pin_data, q_cur).copy()
        tau = tau_pid + g

        # 将控制信号分配到各个 actuator
        for j, actuator in enumerate(self.ACTUATORS):
            self.mj_data.actuator(actuator).ctrl = tau[j]

        # 在一个控制周期内多步更新 Mujoco 状态
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.mj_model, self.mj_data)

    def render(self):
        if self.viewer.is_running():
            self.viewer.sync()

    def print_end_effector_pose(self):
        q_cur = self.mj_data.qpos.copy()
        pin.forwardKinematics(self.pin_model, self.pin_data, q_cur)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        frame_id = self.pin_model.getFrameId("link_tool")
        if frame_id < 0:
            print("末端框架 'link_tool' 未找到！")
            return
        placement = self.pin_data.oMf[frame_id]
        pos = placement.translation
        rpy = R.from_matrix(placement.rotation).as_euler('xyz', degrees=True)
        print("末端位置:", pos)
        print("末端姿态 (roll, pitch, yaw in degrees):", rpy)

# ---------------------------
# DearPyGui 回调函数：读取输入框并更新对应滑块
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
# 主程序入口：集成 GUI 与 Mujoco 仿真
# ---------------------------
def main():
    robot = Robot(control_freq=50)

    # 创建 DearPyGui 上下文，注册较大字号字体（请根据实际系统调整字体路径）
    dpg.create_context()
    with dpg.font_registry():
        default_font = dpg.add_font("C:/Windows/Fonts/Arial.ttf", 20)
    dpg.create_viewport(title="Joint Angle Control (PID)", width=600, height=500)
    dpg.bind_font(default_font)

    # 创建 GUI 窗口：每个关节一行（滑块 + 输入框 + Set 按钮）
    with dpg.window(label="Joint Angle Controls", width=600, height=500):
        for i in range(7):
            dpg.add_text(f"Joint {i+1}")
            dpg.add_slider_float(label="", tag=f"joint_{i+1}_slider",
                                 default_value=0.0, min_value=-3.14, max_value=3.14,
                                 width=300)
            dpg.add_same_line()
            dpg.add_input_text(label="", tag=f"joint_{i+1}_input", default_value="0.0", width=100)
            dpg.add_same_line()
            dpg.add_button(label="Set", callback=set_joint_value,
                           user_data={"slider_tag": f"joint_{i+1}_slider", "input_tag": f"joint_{i+1}_input"})

    dpg.setup_dearpygui()
    dpg.show_viewport()

    step_count = 0
    # 主循环：不断从 GUI 读取期望关节角度，控制机器人运动，同时渲染仿真界面
    while robot.viewer.is_running() and dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()
        joint_angles = []
        for i in range(7):
            angle = dpg.get_value(f"joint_{i+1}_slider")
            joint_angles.append(angle)
        q_des = np.array(joint_angles)
        robot.step(q_des)
        robot.render()

        step_count += 1
        if step_count % 100 == 0:
            robot.print_end_effector_pose()

    dpg.destroy_context()

if __name__ == "__main__":
    main()
