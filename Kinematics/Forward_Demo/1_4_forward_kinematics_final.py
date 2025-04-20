import pinocchio as pin
import numpy as np
import mujoco
from mujoco import viewer
import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation as R

def set_joint_value(sender, app_data, user_data):
    # user_data 包含 slider_tag 与 input_tag
    slider_tag = user_data["slider_tag"]
    input_tag = user_data["input_tag"]
    try:
        # 从输入框获取字符串，并转换为浮点数
        input_str = dpg.get_value(input_tag)
        value = float(input_str)
        dpg.set_value(slider_tag, value)
    except Exception as e:
        print("输入值有误:", e)

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

    # 打印末端位姿信息
    print("末端位置:", pos)
    print("末端姿态 (旋转矩阵):")
    print(rot)
    print("末端姿态 (roll, pitch, yaw in degrees):", rpy_angles)

    # 渲染 Mujoco 窗口
    if mj_viewer.is_running():
        mj_viewer.sync()

def main():
    # 加载 Pinocchio 模型（请根据实际情况修改 URDF 路径）
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

    # 使用 DearPyGui 创建滑块界面，并设置较大字体和较宽的控件
    dpg.create_context()
    with dpg.font_registry():
        default_font = dpg.add_font("C:/Windows/Fonts/Arial.ttf", 40)
    dpg.create_viewport(title="Joint Control Window", width=675, height=850)
    dpg.bind_font(default_font)
    with dpg.window(label="Joint Angle Controls", width=650, height=850):
        # 为每个关节创建一行：滑块、输入框和按钮
        for i in range(7):
            dpg.add_text(f"Joint {i+1}")
            dpg.add_slider_float(label="", tag=f"joint_{i+1}_slider",
                                 default_value=0.0, min_value=-3.14, max_value=3.14,
                                 width=300)
            dpg.add_same_line()
            # 使用 add_input_text 来实现纯文本输入数字
            dpg.add_input_text(label="", tag=f"joint_{i+1}_input", default_value="0.0", width=100)
            dpg.add_same_line()
            dpg.add_button(label="Set", callback=set_joint_value,
                           user_data={"slider_tag": f"joint_{i+1}_slider", "input_tag": f"joint_{i+1}_input"})
    dpg.setup_dearpygui()
    dpg.show_viewport()

    # 主循环：同时更新 GUI 和 Mujoco 仿真
    while mj_viewer.is_running() and dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()
        simulation_loop(mj_model, mj_data, mj_viewer, pin_model, pin_data, frame_id)

    dpg.destroy_context()

if __name__ == "__main__":
    main()
