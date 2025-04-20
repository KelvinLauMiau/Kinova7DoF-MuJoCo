import mujoco
from mujoco import viewer
import numpy as np
import pinocchio as pin
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel
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

    def get_gravity_mat(self, q):
        return pin.computeGeneralizedGravity(self.model, self.data, q).copy()


# ------------------------------
# 机械臂主控制类
# ------------------------------
class Robot:
    ACTUATORS = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7']

    def __init__(self, urdf_path):
        self.mj_model = mujoco.MjModel.from_xml_path('../Model/ActualArm/Kinova_mjmodel.xml')
        self.mj_data = mujoco.MjData(self.mj_model)
        self.viewer = viewer.launch_passive(self.mj_model, self.mj_data)
        self.solver = PinSolver(urdf_path)
        self.external_force = np.zeros(7)

    def apply_external_force(self, force):
        self.external_force = force

    def step(self):
        g = self.solver.get_gravity_mat(self.mj_data.qpos)
        torque = self.external_force + g
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
        self.joystick_fx_fy = Joystick("Fx/Fy")
        self.joystick_fz_tz = Joystick("Fz/Tz")
        self.joystick_tx_ty = Joystick("Tx/Ty")
        layout.addWidget(self.joystick_fx_fy)
        layout.addWidget(self.joystick_fz_tz)
        layout.addWidget(self.joystick_tx_ty)
        self.setLayout(layout)
        self.joystick_fx_fy.joystick_moved.connect(self.update_fx_fy)
        self.joystick_fz_tz.joystick_moved.connect(self.update_fz_tz)
        self.joystick_tx_ty.joystick_moved.connect(self.update_tx_ty)
        self.force = np.zeros(7)

    def update_fx_fy(self, x, y):
        self.force[0] = x * 0.5  # Fx
        self.force[1] = y * 0.5  # Fy
        self.robot.apply_external_force(self.force)

    def update_fz_tz(self, x, y):
        self.force[2] = y * 0.5  # Fz
        self.force[5] = x * 1.0  # Tz
        self.robot.apply_external_force(self.force)

    def update_tx_ty(self, x, y):
        self.force[3] = x * 1.0  # Tx
        self.force[4] = y * 1.0  # Ty
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
