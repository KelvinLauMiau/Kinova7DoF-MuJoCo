import pinocchio as pin
import numpy as np
import mujoco
from mujoco import viewer

def main():
    # 加载 Pinocchio 模型（URDF）
    urdf_path = "../../Kinova_description/urdf/Kinova_description.urdf"
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()

    # 加载 Mujoco 模型（MJCF）
    mjcf_path = "../../Model/ActualArm/Kinova_mjmodel.xml"  # 根据你的文件路径修改
    mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
    mj_data = mujoco.MjData(mj_model)
    mj_viewer = viewer.launch_passive(mj_model, mj_data)

    # 获取末端框架在 Pinocchio 中的 frame_id（假定名称为 "link_tool"）
    frame_id = model.getFrameId("link_tool")
    if frame_id < 0:
        print("末端框架 'link_tool' 未找到，请检查 URDF 文件中的框架名称。")
        return

    print("请按格式输入关节角度，例如：#0.0,0.1,0.2,0.3,0.4,0.5,0.6")
    while True:
        # 获取用户输入
        input_str = input("请输入关节角度 (#j1,j2,j3,j4,j5,j6,j7): ")
        if input_str.startswith('#'):
            input_str = input_str[1:]
        try:
            q_list = list(map(float, input_str.split(',')))
            if len(q_list) != 7:
                raise ValueError("输入的角度数量不为 7")
        except Exception as e:
            print("输入格式错误:", e)
            continue

        q = np.array(q_list)

        # 使用 Pinocchio 计算正向运动学
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        placement = data.oMf[frame_id]
        pos = placement.translation
        rot = placement.rotation

        print("计算得到末端位置:", pos)
        print("计算得到末端姿态 (旋转矩阵):")
        print(rot)

        # 更新 Mujoco 仿真中的关节角度
        # 注意：这里假定 Mujoco 模型的 qpos 仅包含 7 个关节，
        # 若你的模型有其他自由度，需要做对应的调整
        mj_data.qpos[:] = q
        # 更新 Mujoco 状态
        mujoco.mj_forward(mj_model, mj_data)

        # 渲染视图
        if mj_viewer.is_running():
            mj_viewer.sync()

if __name__ == "__main__":
    main()
