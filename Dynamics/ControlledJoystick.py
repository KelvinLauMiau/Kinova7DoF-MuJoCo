import mujoco
from mujoco import viewer
import numpy as np
import pinocchio as pin
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor
from PyQt6.QtCore import Qt, QPointF, pyqtSignal
import sys
import math


# ------------------------------
# 机械臂动力学求解器
# ------------------------------
class PinSolver:
    def __init__(self, urdf_path: str):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self._JOINT_NUM = self.model.nq

    def get_inertia_mat(self, q):
        return pin.crba(self.model, self.data, q).copy()

    def get_coriolis_mat(self, q, qdot):
        return pin.computeCoriolisMatrix(self.model, self.data, q, qdot).copy()

    def get_gravity_mat(self, q):
        return pin.computeGeneralizedGravity(self.model, self.data, q).copy()


# ------------------------------
# 关节阻抗控制器
# ------------------------------
class JntImpedance:
    def __init__(self, urdf_path: str):
        self.kd_solver = PinSolver(urdf_path)
        self.k = 6.0 * np.ones(7)  # 低刚度
        self.B = 0.8 * np.ones(7)  # 低阻尼

    def compute_jnt_torque(self, q_cur, v_cur, external_force):
        M = self.kd_solver.get_inertia_mat(q_cur)
        C = self.kd_solver.get_coriolis_mat(q_cur, v_cur)
        g = self.kd_solver.get_gravity_mat(q_cur)
        tau = np.dot(M, external_force) + C[-1] + g
        return tau


# ------------------------------
# 机械臂主控制类
# ------------------------------
class Robot:
    ACTUATORS = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7']

    def __init__(self, urdf_path, control_freq=20):
        self.mj_model = mujoco.MjModel.from_xml_path('../Model/ActualArm/Kinova_mjmodel.xml')
        self.mj_data = mujoco.MjData(self.mj_model)
        self.viewer = viewer.launch_passive(self.mj_model, self.mj_data)
        self.controller = JntImpedance(urdf_path)
        self.external_force = np.zeros(7)

    def apply_external_force(self, force):
        self.external_force = force

    def step(self):
        torque = self.controller.compute_jnt_torque(self.mj_data.qpos, self.mj_data.qvel, self.external_force)
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

        # 画背景
        painter.setPen(QPen(Qt.GlobalColor.black, 3))
        painter.setBrush(QBrush(QColor(200, 200, 200)))
        painter.drawEllipse(50, 50, 100, 100)

        # 画摇杆
        painter.setPen(QPen(Qt.GlobalColor.blue, 2))
        painter.setBrush(QBrush(QColor(0, 0, 255)))
        painter.drawEllipse(self.joystick_pos, 20, 20)

    def mousePressEvent(self, event):
        if (event.position() - self.joystick_center).manhattanLength() <= self.radius:
            self.dragging = True

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

        self.joystick_force = Joystick()
        self.joystick_torque = Joystick()

        layout.addWidget(self.joystick_force)
        layout.addWidget(self.joystick_torque)
        self.setLayout(layout)

        self.joystick_force.joystick_moved.connect(self.update_force)
        self.joystick_torque.joystick_moved.connect(self.update_torque)

        self.force = np.zeros(7)

    def update_force(self, x, y):
        self.force[0] = x * 1.0  # Fx
        self.force[1] = y * 1.0  # Fy
        self.robot.apply_external_force(self.force)

    def update_torque(self, x, y):
        self.force[3] = x * 10.0  # Mx
        self.force[4] = y * 10.0  # My
        self.robot.apply_external_force(self.force)


# ------------------------------
# 主程序
# ------------------------------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    urdf_path = '../Kinova_description/urdf/Kinova_description.urdf'
    robot = Robot(urdf_path)
    joystick_control = JoystickController(robot)
    joystick_control.show()

    while True:
        robot.step()
        robot.render()
        app.processEvents()
