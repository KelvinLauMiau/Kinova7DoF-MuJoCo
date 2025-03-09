import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares

class IKOptimizer:
    """
    利用 SciPy 的 least_squares 求解 IK 问题：
      - forward_kinematics(q): 给定关节角度 q，返回末端位姿（SE3 对象）。
      - objective(q, target_pose): 计算正向运动学误差向量 f(q) = log(FK(q)⁻¹ * target_pose)。
      - solve(target_pose, q0): 以 q0 为初始猜测求解 IK，关节角度边界为 [-3.14, 3.14]。
    """

    def __init__(self, urdf_path: str, ee_frame_name: str):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.ee_frame_id = self.model.getFrameId(ee_frame_name)
        if self.ee_frame_id < 0:
            raise ValueError(f"未找到末端框架 {ee_frame_name}，请检查 URDF 文件！")

    def forward_kinematics(self, q):
        q = np.asarray(q, dtype=np.float64)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.ee_frame_id]

    def objective(self, q, target_pose):
        # 计算当前末端位姿
        current_pose = self.forward_kinematics(q)
        # 计算 6 维误差向量
        error = pin.log(current_pose.inverse() * target_pose)
        return error

    def solve(self, target_pose, q0):
        # 设置关节角度边界：每个关节都限制在 [-3.14, 3.14]
        lb = np.full_like(q0, -3.14)
        ub = np.full_like(q0,  3.14)
        res = least_squares(self.objective, q0, args=(target_pose,), bounds=(lb, ub), verbose=2)
        return res.x

if __name__ == "__main__":
    # 修改 URDF 路径和末端框架名称，确保与你的文件一致
    urdf_path = "../../Kinova_description/urdf/Kinova_description.urdf"
    ee_frame_name = "link_tool"
    ik_optimizer = IKOptimizer(urdf_path, ee_frame_name)

    # 初始猜测：这里用全零作为示例，实际可用当前状态
    q0 = np.zeros(7)

    # 首先计算当前末端位姿
    current_pose = ik_optimizer.forward_kinematics(q0)
    print("当前末端位姿:")
    print("位置:", current_pose.translation)
    print("旋转矩阵:")
    print(current_pose.rotation)
    rpy_current = R.from_matrix(current_pose.rotation).as_euler('xyz', degrees=True)
    print("欧拉角 (roll, pitch, yaw in degrees):", rpy_current)

    # 构造目标末端位姿：
    # 期望末端位置: [-0.989, -0.021, 0.581]
    # 期望末端旋转矩阵:
    # [[ 1.52655509e-06,  9.99999684e-01,  7.95491984e-04],
    #  [-9.99999746e-01,  9.59288810e-07,  7.13101082e-04],
    #  [ 7.13100093e-04, -7.95492870e-04,  9.99999429e-01]]
    target_translation = np.array([-0.989, -0.021, 0.581])
    target_rotation = np.array([[1.52655509e-06,  9.99999684e-01,  7.95491984e-04],
                                [-9.99999746e-01,  9.59288810e-07,  7.13101082e-04],
                                [7.13100093e-04, -7.95492870e-04,  9.99999429e-01]])
    target_pose = pin.SE3(target_rotation, target_translation)
    print("\n目标末端位姿:")
    print("位置:", target_pose.translation)
    print("旋转矩阵:")
    print(target_pose.rotation)
    rpy_target = R.from_matrix(target_pose.rotation).as_euler('xyz', degrees=True)
    print("欧拉角 (roll, pitch, yaw in degrees):", rpy_target)

    # 调用 least_squares 求解 IK
    q_opt = ik_optimizer.solve(target_pose, q0)
    print("\n求解得到的关节角度:", q_opt)

    # 利用正向运动学验证 IK 结果
    fk_pose = ik_optimizer.forward_kinematics(q_opt)
    print("\n正向运动学验证得到的末端位姿:")
    print("位置:", fk_pose.translation)
    print("旋转矩阵:")
    print(fk_pose.rotation)
    rpy_fk = R.from_matrix(fk_pose.rotation).as_euler('xyz', degrees=True)
    print("欧拉角 (roll, pitch, yaw in degrees):", rpy_fk)
