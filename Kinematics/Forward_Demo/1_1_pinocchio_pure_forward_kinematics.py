import pinocchio as pin
import numpy as np

def main():
    # 请确保这里的 URDF 路径正确
    urdf_path = "../../Kinova_description/urdf/Kinova_description.urdf"
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()

    # 假设末端框架名称为 "link_tool"
    frame_id = model.getFrameId("link_tool")
    if frame_id < 0:
        print("末端框架 'link_tool' 未找到，请检查 URDF 文件中的框架名称。")
        return

    # 从 terminal 输入关节角度，例如 "#0.0,0.1,0.2,0.3,0.4,0.5,0.6"
    input_str = input("请输入关节角度 (#j1,j2,j3,j4,j5,j6,j7): ")
    if input_str.startswith('#'):
        input_str = input_str[1:]
    try:
        q_list = list(map(float, input_str.split(',')))
        if len(q_list) != 7:
            raise ValueError("输入的角度数量不为 7")
    except Exception as e:
        print("输入格式错误:", e)
        return

    q = np.array(q_list)

    # 计算正向运动学
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    # 获取末端框架的位置与旋转矩阵
    placement = data.oMf[frame_id]
    pos = placement.translation
    rot = placement.rotation

    print("末端位置:", pos)
    print("末端姿态 (旋转矩阵):")
    print(rot)

if __name__ == "__main__":
    while True:
        main()
