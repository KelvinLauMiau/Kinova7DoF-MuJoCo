import numpy as np
import pinocchio as pin
import mujoco
from mujoco import viewer
from scipy.spatial.transform import Rotation as R
import dearpygui.dearpygui as dpg

# --------------------------------------------------
# Pinocchio 动力学/运动学求解器（原有代码）
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
# 关节阻抗控制器（原有代码）
# --------------------------------------------------
class JntImpedance:
    def __init__(self, urdf_path: str):
        self.kd_solver = PinSolver(urdf_path)
        self.k = 6.0 * np.ones(7)
        self.B = 1.5 * np.ones(7)

    def compute_jnt_torque(self, q_des, v_des, q_cur, v_cur):
        M = self.kd_solver.get_inertia_mat(q_cur)
        C = self.kd_solver.get_coriolis_mat(q_cur, v_cur)
        g = self.kd_solver.get_gravity_mat(q_cur)
        # 采用原代码中的写法：C[-1] + g
        coriolis_gravity = C[-1] + g
        acc_desire = self.k * (q_des - q_cur) + self.B * (v_des - v_cur)
        tau = np.dot(M, acc_desire) + coriolis_gravity
        return tau

# --------------------------------------------------
# Drag Teaching Forward_Demo 的机器人类（集成了阻抗控制）
# --------------------------------------------------
class Robot:
    ACTUATORS = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7']

    def __init__(self, control_freq=50):
        # 加载 Mujoco 模型（MJCF 文件），请根据实际情况修改路径
        self.mj_model = mujoco.MjModel.from_xml_path("../../Model/Kinova_mjmodel.xml")
        self.mj_data = mujoco.MjData(self.mj_model)
        self.viewer = viewer.launch_passive(self.mj_model, self.mj_data)

        # 根据控制频率计算每个控制周期内的步数
        control_timestep = 1.0 / control_freq
        model_timestep = self.mj_model.opt.timestep
        self._n_substeps = int(control_timestep / model_timestep)

        # 初始化阻抗控制器，URDF 路径请根据实际情况修改
        self.controller = JntImpedance(urdf_path="../../Kinova_description/urdf/Kinova_description.urdf")

    def step(self, action: np.ndarray):
        # 在一个控制周期内进行多次模型步进
        for _ in range(self._n_substeps):
            self.inner_step(action)
            mujoco.mj_step(self.mj_model, self.mj_data)

    def inner_step(self, action):
        # 根据阻抗控制器计算期望力矩
        torque = self.controller.compute_jnt_torque(
            q_des=action,
            v_des=np.zeros(7),
            q_cur=self.mj_data.qpos,
            v_cur=self.mj_data.qvel,
        )
        # 将计算得到的力矩分配给各个 actuator
        for j, actuator_name in enumerate(self.ACTUATORS):
            self.mj_data.actuator(actuator_name).ctrl = torque[j]

    def render(self):
        if self.viewer.is_running():
            self.viewer.sync()

    def print_end_effector_pose(self):
        """
        利用 Pinocchio 计算正向运动学，并打印末端框架（假定名称为 "link_tool"）的位置及姿态（roll, pitch, yaw，角度制）
        """
        solver = self.controller.kd_solver
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
        rpy = R.from_matrix(rot).as_euler('xyz', degrees=True)
        print("末端位置:", pos)
        print("末端姿态 (roll, pitch, yaw in degrees):", rpy)

# --------------------------------------------------
# DearPyGui 的回调函数：从输入文本框获取数值并更新对应滑块
# --------------------------------------------------
def set_joint_value(sender, app_data, user_data):
    slider_tag = user_data["slider_tag"]
    input_tag = user_data["input_tag"]
    try:
        input_str = dpg.get_value(input_tag)
        value = float(input_str)
        dpg.set_value(slider_tag, value)
    except Exception as e:
        print("输入值有误:", e)

# --------------------------------------------------
# 主程序：集成拖曳教学 demo 与 GUI 控制
# --------------------------------------------------
def main():
    # 初始化机器人仿真
    robot = Robot(control_freq=50)

    # 创建 DearPyGui 上下文，并注册字体（根据操作系统调整字体路径）
    dpg.create_context()
    with dpg.font_registry():
        default_font = dpg.add_font("C:/Windows/Fonts/Arial.ttf", 20)
    dpg.create_viewport(title="Joint Angle Control", width=600, height=500)
    dpg.bind_font(default_font)

    # 创建 GUI 窗口：每个关节一行，包含滑块、文本输入框及 Set 按钮
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
    # 主循环：同时更新 GUI 与 Mujoco 仿真
    while robot.viewer.is_running() and dpg.is_dearpygui_running():
        # 渲染 GUI 界面
        dpg.render_dearpygui_frame()

        # 从 GUI 滑块中读取每个关节的角度
        joint_angles = []
        for i in range(7):
            angle = dpg.get_value(f"joint_{i+1}_slider")
            joint_angles.append(angle)
        q_des = np.array(joint_angles)

        # 用 GUI 中的关节角度作为期望关节角度，执行一个控制周期
        robot.step(q_des)
        robot.render()

        # 每隔 100 步打印一次末端位姿
        step_count += 1
        if step_count % 100 == 0:
            robot.print_end_effector_pose()

    dpg.destroy_context()

if __name__ == "__main__":
    main()
