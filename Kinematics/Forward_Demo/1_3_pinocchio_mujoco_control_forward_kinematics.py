import pinocchio as pin
import numpy as np
import mujoco
from mujoco import viewer
import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation as R

def simulation_loop(mj_model, mj_data, mj_viewer, pin_model, pin_data, frame_id):
    # 从每个滑块获取当前关节角度值
    joint_angles = []
    for i in range(7):
        value = dpg.get_value(f"joint_{i+1}_slider")
        joint_angles.append(value)
    q = np.array(joint_angles)

    # 更新 Mujoco 仿真中的关节状态
    mj_data.qpos[:] = q
    mujoco.mj_forward(mj_model, mj_data)

    # 使用 Pinocchio 计算正向运动学
    pin.forwardKinematics(pin_model, pin_data, q)
    pin.updateFramePlacements(pin_model, pin_data)
    placement = pin_data.oMf[frame_id]
    pos = placement.translation
    rot = placement.rotation

    # 利用 SciPy 将旋转矩阵转换为 roll, pitch, yaw（角度制）
    rpy_angles = R.from_matrix(rot).as_euler('xyz', degrees=True)

    # 将末端位姿打印到 terminal
    print("末端位置:", pos)
    print("末端姿态 (旋转矩阵):", rot)
    print("末端姿态 (roll, pitch, yaw in degrees):", rpy_angles)

    # 渲染 Mujoco 窗口
    if mj_viewer.is_running():
        mj_viewer.sync()

def main():
    # 加载 Pinocchio 模型（URDF 路径请根据实际情况修改）
    urdf_path = "../../Kinova_description/urdf/Kinova_description.urdf"
    pin_model = pin.buildModelFromUrdf(urdf_path)
    pin_data = pin_model.createData()

    # 加载 Mujoco 模型（MJCF 文件路径）
    mjcf_path = "../../Model/ActualArm/Kinova_mjmodel.xml"  # 请确保路径正确
    mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
    mj_data = mujoco.MjData(mj_model)
    mj_viewer = viewer.launch_passive(mj_model, mj_data)

    # 获取末端框架在 Pinocchio 模型中的 id（假设名称为 "link_tool"）
    frame_id = pin_model.getFrameId("link_tool")
    if frame_id < 0:
        print("末端框架 'link_tool' 未找到，请检查 URDF 文件中的框架名称。")
        return

    # 使用 DearPyGui 创建滑块界面
    dpg.create_context()

    # 注册字体（请根据实际系统路径修改字体文件路径）
    with dpg.font_registry():
        default_font = dpg.add_font("C:/Windows/Fonts/Arial.ttf", 48)

    dpg.create_viewport(title="Joint Angle Control", width=600, height=550)
    dpg.bind_font(default_font)

    with dpg.window(label="Joint Angle Sliders", width=600, height=550):
        # 为每个关节添加一个滑块，设置较宽的宽度（例如300）
        for i in range(7):
            dpg.add_slider_float(label=f"Joint {i+1}", tag=f"joint_{i+1}_slider",
                                 default_value=0.0, min_value=-3.14, max_value=3.14,
                                 width=300)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    # 主循环：同时更新 GUI 和 Mujoco 仿真
    while mj_viewer.is_running() and dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()
        simulation_loop(mj_model, mj_data, mj_viewer, pin_model, pin_data, frame_id)

    dpg.destroy_context()

if __name__ == "__main__":
    main()
