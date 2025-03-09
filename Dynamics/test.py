import mujoco
from mujoco import viewer
import numpy as np
import pinocchio as pin
import dearpygui.dearpygui as dpg

# ==================================================
# PinSolver：利用 Pinocchio 加载 URDF 模型并计算动力学量
# ==================================================
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

# ==================================================
# JntImpedance：阻抗控制器（拖动示教用，参数 k=6.0, B=0.8）
# ==================================================
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

# ==================================================
# Robot：拖动示教 Demo（目标控制器为拖动示教方式），仅针对 Target Arm 部分
# ==================================================
class Robot:
    # 这里假设你的 MJCF 文件中 Target Arm 部分的 actuators 名称为 a1～a7，
    # 而你希望控制的对象是 link_tool（在 link7 内部的虚拟 body）
    ACTUATORS = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7']
    def __init__(self, control_freq=20) -> None:
        self.mj_model = mujoco.MjModel.from_xml_path(filename='../Model/WithTool/KinovaGen3_7dof_description/Kinova_tool.xml')
        self.mj_data = mujoco.MjData(self.mj_model)
        self.viewer = viewer.launch_passive(self.mj_model, self.mj_data)
        control_timestep = 1.0 / control_freq
        model_timestep = self.mj_model.opt.timestep
        self._n_substeps = int(control_timestep / model_timestep)
        self.controller = JntImpedance(urdf_path='../Kinova_description/urdf/Kinova_description.urdf')
        # 初始目标角度设为当前状态（假设整个 qpos 保存拖动示教的目标）
        self.q_des = self.mj_data.qpos.copy()

    def render(self):
        if self.viewer.is_running():
            self.viewer.sync()

    def step(self, action: np.ndarray):
        for _ in range(self._n_substeps):
            self.inner_step(action)
            mujoco.mj_step(self.mj_model, self.mj_data)

    def inner_step(self, action):
        torque = self.controller.compute_jnt_torque(
            q_des=action,
            v_des=np.zeros(7),
            q_cur=self.mj_data.qpos,
            v_cur=self.mj_data.qvel,
        )
        for j, act in enumerate(self.ACTUATORS):
            self.mj_data.actuator(act).ctrl = torque[j]

# ==================================================
# GUI：外力控制（施加到虚拟 body "link_tool"）——滑块范围调为 -10 到 10
# ==================================================
def create_force_gui():
    with dpg.window(label="External Force Control", pos=(600, 0), width=400, height=300):
        dpg.add_checkbox(label="Enable External Force", tag="force_switch", default_value=False)
        dpg.add_slider_float(label="Force X", tag="force_x", default_value=0.0, min_value=-10, max_value=10)
        dpg.add_slider_float(label="Force Y", tag="force_y", default_value=0.0, min_value=-10, max_value=10)
        dpg.add_slider_float(label="Force Z", tag="force_z", default_value=0.0, min_value=-10, max_value=10)
        dpg.add_slider_float(label="Torque X", tag="torque_x", default_value=0.0, min_value=-10, max_value=10)
        dpg.add_slider_float(label="Torque Y", tag="torque_y", default_value=0.0, min_value=-10, max_value=10)
        dpg.add_slider_float(label="Torque Z", tag="torque_z", default_value=0.0, min_value=-10, max_value=10)

# ==================================================
# 主函数：整合拖动示教与外力控制
# ==================================================
def main():
    dpg.create_context()
    create_force_gui()
    dpg.create_viewport(title="External Force Control", width=400, height=300)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    robot = Robot(control_freq=20)
    # 获取虚拟 body "link_tool" 的 id（从 MJCF 中查找）
    body_id = mujoco.mj_name2id(robot.mj_model, mujoco.mjtObj.mjOBJ_BODY, "link_tool")
    if body_id < 0:
        print("未找到 body 'link_tool'!")
    external_force = np.zeros(6)
    while robot.viewer.is_running() and dpg.is_dearpygui_running():
        # 更新外力控制
        if dpg.get_value("force_switch"):
            fx = dpg.get_value("force_x")
            fy = dpg.get_value("force_y")
            fz = dpg.get_value("force_z")
            tx = dpg.get_value("torque_x")
            ty = dpg.get_value("torque_y")
            tz = dpg.get_value("torque_z")
            external_force = np.array([fx, fy, fz, tx, ty, tz])
        # 始终施加当前 external_force 到虚拟 body "link_tool"
        robot.mj_data.xfrc_applied[body_id] = external_force

        # 拖动示教：目标角度保持 robot.q_des（初始为程序启动时的状态），
        # 拖动后 qpos 偏离目标时，控制器产生恢复力矩将其拉回
        robot.step(robot.mj_data.qpos)
        robot.render()
        dpg.render_dearpygui_frame()
    dpg.destroy_context()

if __name__ == "__main__":
    main()
