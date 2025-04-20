import mujoco
from mujoco import viewer
import numpy as np
import pinocchio as pin
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor
from PyQt6.QtCore import Qt, QPointF, pyqtSignal
import sys
import math


# ------------------------------
# PD 控制器
# ------------------------------
class PDController:
    def __init__(self, kp=10.0, kd=1.0):
        self.kp = kp  # 比例增益
        self.kd = kd  # 微分增益
        self.last_error = np.zeros(6)

    def compute(self, target, current):
        """计算 PD 控制力/力矩输出"""
        error = target - current
        d_error = error - self.last_error
        self.last_error = error
        return self.kp * error + self.kd * d_error


# ------------------------------
# 机械臂动力学求解器
# ------------------------------
class PinSolver:
    def __init__(self, urdf_path: str):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self._JOINT_NUM = self.model.nq

    def get_gravity_mat(self, q):
        return pin.computeGeneralizedGravity(self.model, self.data, q).copy()


# ------------------------------
# 机械臂主控制类
# ------------------------------
class Robot:
    ACTUATORS = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7']

    def __init__(self, urdf_path, kp=10.0, kd=1.0):
        self.mj_model = mujoco.MjModel.from_xml_path('../Model/ActualArm/Kinova_mjmodel.xml')
        self.mj_data = mujoco.MjData(self.mj_model)
        self.viewer = viewer.launch_passive(self.mj_model, self.mj_data)
        self.solver = PinSolver(urdf_path)

        # PD 控制器
        self.pd_controller = PDController(kp, kd)
        self.target_pose = np.zeros(6)  # 目标末端位置+姿态 (x, y, z, roll, pitch, yaw)

    def set_target_pose(self, pose):
        """设置机械臂的目标位置和姿态"""
        self.target_pose = pose

    def get_current_pose(self):
        """计算当前末端的位置和姿态"""
        pin.forwardKinematics(self.solver.model, self.solver.data, self.mj_data.qpos)
        pin.updateFramePlacements(self.solver.model, self.solver.data)

        # 获取机械臂末端位姿
        end_effector_pose = self.solver.data.oMf[-1]
        position = end_effector_pose.translation
        rotation_matrix = end_effector_pose.rotation
        rpy = pin.rpy.matrixToRpy(rotation_matrix)

        return np.concatenate([position, rpy])  # 6维向量 (x, y, z, roll, pitch, yaw)

    def step(self):
        """执行控制循环"""
        current_pose = self.get_current_pose()
        control_force = self.pd_controller.compute(self.target_pose, current_pose)

        # 扩展为 7 维，使其与关节数量匹配
        control_force_extended = np.zeros(7)
        control_force_extended[:6] = control_force  # 仅控制前6个关节，最后一个关节设为0

        # 应用控制力和力矩
        g = self.solver.get_gravity_mat(self.mj_data.qpos)
        torque = control_force_extended + g  # 确保两者维度一致

        for j, per_actuator_index in enumerate(self.ACTUATORS):
            self.mj_data.actuator(per_actuator_index).ctrl = torque[j]

        mujoco.mj_step(self.mj_model, self.mj_data)

    def render(self):
        if self.viewer.is_running():
            self.viewer.sync()


# ------------------------------
# 摇杆控制器
# ------------------------------
class Joystick(QWidget):
    joystick_moved = pyqtSignal(float, float)

    def __init__(self, label_text):
        super().__init__()
        self.setFixedSize(200, 200)
        self.joystick_center = QPointF(100, 100)
        self.joystick_pos = QPointF(100, 100)
        self.radius = 40
        self.dragging = False
        self.label = QLabel(label_text, self)
        self.label.move(60, 170)

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
        if self.dragging:
            diff = event.position() - self.joystick_center
            dist = min(math.sqrt(diff.x() ** 2 + diff.y() ** 2), self.radius)
            angle = math.atan2(diff.y(), diff.x())
            self.joystick_pos = QPointF(
                self.joystick_center.x() + dist * math.cos(angle),
                self.joystick_center.y() + dist * math.sin(angle)
            )
            self.update()
            x_norm = (self.joystick_pos.x() - self.joystick_center.x()) / self.radius
            y_norm = (self.joystick_pos.y() - self.joystick_center.y()) / self.radius
            self.joystick_moved.emit(x_norm, y_norm)

    def mousePressEvent(self, event):
        self.dragging = True

    def mouseReleaseEvent(self, event):
        self.dragging = False
        self.joystick_pos = self.joystick_center
        self.update()
        self.joystick_moved.emit(0.0, 0.0)


class JoystickController(QWidget):
    def __init__(self, robot):
        super().__init__()
        self.robot = robot
        layout = QVBoxLayout()
        self.joystick_x_y = Joystick("X/Y")
        self.joystick_z_yaw = Joystick("Z/Yaw")
        self.joystick_roll_pitch = Joystick("Roll/Pitch")
        layout.addWidget(self.joystick_x_y)
        layout.addWidget(self.joystick_z_yaw)
        layout.addWidget(self.joystick_roll_pitch)
        self.setLayout(layout)
        self.joystick_x_y.joystick_moved.connect(self.update_x_y)
        self.joystick_z_yaw.joystick_moved.connect(self.update_z_yaw)
        self.joystick_roll_pitch.joystick_moved.connect(self.update_roll_pitch)
        self.target_pose = np.zeros(6)

    def update_x_y(self, x, y):
        self.target_pose[0] = x * 0.2  # X轴
        self.target_pose[1] = y * 0.2  # Y轴
        self.robot.set_target_pose(self.target_pose)

    def update_z_yaw(self, x, y):
        self.target_pose[2] = y * 0.2  # Z轴
        self.target_pose[5] = x * 0.5  # 偏航角
        self.robot.set_target_pose(self.target_pose)

    def update_roll_pitch(self, x, y):
        self.target_pose[3] = x * 0.5  # 翻滚角
        self.target_pose[4] = y * 0.5  # 俯仰角
        self.robot.set_target_pose(self.target_pose)


# ------------------------------
# 主程序
# ------------------------------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    urdf_path = '../Kinova_description/urdf/Kinova_description.urdf'
    robot = Robot(urdf_path, kp=15.0, kd=2.0)
    joystick_control = JoystickController(robot)
    joystick_control.show()

    while True:
        robot.step()
        robot.render()
        app.processEvents()
