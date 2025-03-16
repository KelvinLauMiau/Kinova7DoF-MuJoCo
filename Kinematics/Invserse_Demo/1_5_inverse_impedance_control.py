import sys
import numpy as np
import pinocchio as pin
import mujoco
from mujoco import viewer
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor
from PyQt6.QtCore import Qt, QPointF, QTimer, pyqtSignal
import math


# ------------------------------
# 逆运动学求解器（使用 Pinocchio）
# ------------------------------
class IKOptimizer:
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
        current_pose = self.forward_kinematics(q)
        error = pin.log(current_pose.inverse() * target_pose)
        return error

    def solve(self, target_pose, q0):
        lb = np.full_like(q0, -3.14)
        ub = np.full_like(q0, 3.14)
        res = least_squares(self.objective, q0, args=(target_pose,), bounds=(lb, ub))
        return res.x


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
        # 尝试调整刚度和阻尼参数
        self.k = 40.0 * np.ones(7)
        self.B = 10.0 * np.ones(7)

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


# ------------------------------
# PyQt6 摇杆控制（自动回中）
# ------------------------------
class Joystick(QWidget):
    joystick_moved = pyqtSignal(float, float)

    def __init__(self):
        super().__init__()
        self.setFixedSize(200, 200)
        self.joystick_center = QPointF(100, 100)
        self.joystick_pos = QPointF(100, 100)
        self.radius = 40
        self.dragging = False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(Qt.GlobalColor.black, 3))
        painter.setBrush(QBrush(QColor(200, 200, 200)))
        painter.drawEllipse(50, 50, 100, 100)
        painter.setPen(QPen(Qt.GlobalColor.blue, 2))
        painter.setBrush(QBrush(QColor(0, 0, 255)))
        painter.drawEllipse(self.joystick_pos, 20, 20)

    def mouseMoveEvent(self, event):
        diff = event.position() - self.joystick_center
        dist = min(math.sqrt(diff.x() ** 2 + diff.y() ** 2), self.radius)
        angle = math.atan2(diff.y(), diff.x())
        self.joystick_pos = QPointF(
            self.joystick_center.x() + dist * math.cos(angle),
            self.joystick_center.y() + dist * math.sin(angle)
        )
        self.update()
        self.joystick_moved.emit(diff.x() / self.radius, diff.y() / self.radius)

    def mouseReleaseEvent(self, event):
        """释放鼠标后自动回中"""
        self.joystick_pos = self.joystick_center
        self.update()
        self.joystick_moved.emit(0.0, 0.0)


# ------------------------------
# 摇杆控制
# ------------------------------
class JoystickControl(QWidget):
    def __init__(self, robot):
        super().__init__()
        self.robot = robot
        layout = QVBoxLayout()
        joystick_layout = QHBoxLayout()

        # ✅ 第一个摇杆：控制 X 和 Y
        self.joystick_xy = Joystick()
        joystick_layout.addWidget(self.joystick_xy)

        # ✅ 第二个摇杆：控制 Z 和 Z 旋转
        self.joystick_z_rz = Joystick()
        joystick_layout.addWidget(self.joystick_z_rz)

        # ✅ 第三个摇杆：控制 X 旋转 和 Y 旋转
        self.joystick_rx_ry = Joystick()
        joystick_layout.addWidget(self.joystick_rx_ry)

        layout.addLayout(joystick_layout)
        self.setLayout(layout)

        # 绑定信号到更新方法
        self.joystick_xy.joystick_moved.connect(self.update_xy_position)
        self.joystick_z_rz.joystick_moved.connect(self.update_z_rz_position)
        self.joystick_rx_ry.joystick_moved.connect(self.update_rx_ry_position)

        # 初始化增量变量
        self.delta_x = 0.0
        self.delta_y = 0.0
        self.delta_z = 0.0
        self.delta_rx = 0.0
        self.delta_ry = 0.0
        self.delta_rz = 0.0

        # 计时器控制实时更新
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_robot_position)
        self.timer.start(100)

        # 确保窗口显示
        self.show()

    def update_xy_position(self, x, y):
        """更新 X 和 Y 轴的增量"""
        self.delta_x = x * 0.05
        self.delta_y = y * 0.05

    def update_z_rz_position(self, x, y):
        """更新 Z 轴平移 和 Z 轴旋转"""
        self.delta_z = y * 0.05
        self.delta_rz = x * 0.1  # 旋转角度（弧度）

    def update_rx_ry_position(self, x, y):
        """更新 X 轴旋转 和 Y 轴旋转"""
        self.delta_rx = y * 0.1  # 旋转角度（弧度）
        self.delta_ry = x * 0.1  # 旋转角度（弧度）

    def update_robot_position(self):
        """更新机械臂位置"""
        self.robot.move_to_position(
            self.delta_x, self.delta_y, self.delta_z,
            self.delta_rx, self.delta_ry, self.delta_rz
        )


# --------------------------------------------------
# Robot：集成逆运动学求解和阻抗控制
# --------------------------------------------------
class Robot:
    ACTUATORS = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7']

    def __init__(self, control_freq=50, mjcf_path="../../Model/ActualArm/Kinova_mjmodel.xml",
                 urdf_path="../../Kinova_description/urdf/Kinova_description.urdf", ee_frame_name="link_tool"):
        # 加载 Mujoco 模型
        self.mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.viewer = viewer.launch_passive(self.mj_model, self.mj_data)

        # 根据控制频率计算每个控制周期内的步数
        self.control_timestep = 1.0 / control_freq
        model_timestep = self.mj_model.opt.timestep
        self.n_substeps = int(self.control_timestep / model_timestep)

        # 初始化逆运动学求解器
        self.ik_solver = IKOptimizer(urdf_path, ee_frame_name)
        self.q_current = np.array([1.57, 0.785, 0.0, 0.785, 0.0, 0.0, -1.57])  # ✅ 更新关节角度
        self.target_translation = np.array([0.021, -0.989, 0.582])  # ✅ 更新初始末端位置
        self.target_rotation = np.eye(3)  # ✅ 更新初始末端旋转矩阵

        # 初始化阻抗控制器（PD形式）
        self.controller = JntImpedance(urdf_path)

    def move_to_position(self, delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz):
        """更新目标位姿"""
        self.target_translation += np.array([delta_x, delta_y, delta_z])

        # 旋转矩阵更新
        rot_x = R.from_euler('x', delta_rx, degrees=False).as_matrix()
        rot_y = R.from_euler('y', delta_ry, degrees=False).as_matrix()
        rot_z = R.from_euler('z', delta_rz, degrees=False).as_matrix()
        self.target_rotation = self.target_rotation @ rot_x @ rot_y @ rot_z

        target_pose = pin.SE3(self.target_rotation, self.target_translation)
        q_new = self.ik_solver.solve(target_pose, self.q_current)
        print("逆运动学求解得到的关节角度 q_new:", q_new)  # 添加打印语句检查 q_new
        self.q_current = q_new
        self.mj_data.qpos[:] = q_new

        # 阻抗控制
        self.step(q_new)

        mujoco.mj_step(self.mj_model, self.mj_data)
        self.viewer.sync()

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


# ------------------------------
# 运行程序
# ------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)

    robot = Robot(control_freq=30,
                  mjcf_path="../../Model/ActualArm/Kinova_mjmodel.xml",
                  urdf_path="../../Kinova_description/urdf/Kinova_description.urdf",
                  ee_frame_name="link_tool")

    joystick_control = JoystickControl(robot)
    # 进入事件循环
    sys.exit(app.exec())