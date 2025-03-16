import sys
import numpy as np
import pinocchio as pin
import mujoco
from mujoco import viewer
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel
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
        ub = np.full_like(q0,  3.14)
        res = least_squares(self.objective, q0, args=(target_pose,), bounds=(lb, ub))
        return res.x

# ------------------------------
# 机械臂控制类（Mujoco）
# ------------------------------
class Robot:
    def __init__(self, mjcf_path, urdf_path, ik_solver):
        self.mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.viewer = viewer.launch_passive(self.mj_model, self.mj_data)
        self.ik_solver = ik_solver
        self.q_current = np.zeros(7)  # 初始关节角度
        self.target_translation = np.array([0.0, 0.0, 0.0])

    def move_to_position(self, delta_x, delta_y, delta_z):
        self.target_translation += np.array([delta_x, delta_y, delta_z])
        target_pose = pin.SE3(np.eye(3), self.target_translation)
        q_new = self.ik_solver.solve(target_pose, self.q_current)
        self.q_current = q_new
        self.mj_data.qpos[:] = q_new
        mujoco.mj_step(self.mj_model, self.mj_data)
        self.viewer.sync()

# ------------------------------
# PyQt6 摇杆控制
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

    def mousePressEvent(self, event):
        if (event.position() - self.joystick_center).manhattanLength() <= self.radius:
            self.dragging = True

    def mouseMoveEvent(self, event):
        if self.dragging:
            diff = event.position() - self.joystick_center
            dist = min(math.sqrt(diff.x()**2 + diff.y()**2), self.radius)
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

class JoystickControl(QWidget):
    def __init__(self, robot):
        super().__init__()
        self.robot = robot
        self.initUI()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_robot_position)
        self.timer.start(100)

    def initUI(self):
        layout = QVBoxLayout()
        joystick_layout = QHBoxLayout()
        self.joystick = Joystick()
        joystick_layout.addWidget(self.joystick)
        layout.addLayout(joystick_layout)
        self.setLayout(layout)
        self.joystick.joystick_moved.connect(self.update_position)
        self.delta_x = 0.0
        self.delta_y = 0.0

    def update_position(self, x, y):
        self.delta_x = x * 0.05
        self.delta_y = y * 0.05

    def update_robot_position(self):
        self.robot.move_to_position(self.delta_x, self.delta_y, 0)

# ------------------------------
# 运行程序
# ------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ik_solver = IKOptimizer("../../Kinova_description/urdf/Kinova_description.urdf", "link_tool")
    robot = Robot("../../Model/ActualArm/Kinova_mjmodel.xml", "../../Kinova_description/urdf/Kinova_description.urdf", ik_solver)
    joystick_control = JoystickControl(robot)
    joystick_control.show()
    sys.exit(app.exec())
