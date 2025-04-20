from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor
from PyQt6.QtCore import Qt, QPointF, pyqtSignal
import sys
import math

MAX_FORCE = 1.0  # 最大线性力 (N)
MAX_TORQUE = 10.0  # 最大力矩 (Nm)

class Joystick(QWidget):
    joystick_moved = pyqtSignal(float, float)  # 发送 X/Y 归一化值

    def __init__(self):
        super().__init__()
        self.setFixedSize(200, 200)
        self.joystick_center = QPointF(100, 100)
        self.joystick_pos = QPointF(100, 100)
        self.radius = 40  # 摇杆最大运动范围
        self.dragging = False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 背景圆圈
        painter.setPen(QPen(Qt.GlobalColor.black, 3))
        painter.setBrush(QBrush(QColor(200, 200, 200)))
        painter.drawEllipse(50, 50, 100, 100)

        # 摇杆
        painter.setPen(QPen(Qt.GlobalColor.blue, 2))
        painter.setBrush(QBrush(QColor(0, 0, 255)))
        painter.drawEllipse(self.joystick_pos, 20, 20)

    def mousePressEvent(self, event):
        if (event.position() - self.joystick_center).manhattanLength() <= self.radius:
            self.dragging = True

    def mouseMoveEvent(self, event):
        if self.dragging:
            # 计算新位置，并限制在圆形范围内
            diff = event.position() - self.joystick_center
            dist = min(math.sqrt(diff.x()**2 + diff.y()**2), self.radius)
            angle = math.atan2(diff.y(), diff.x())
            self.joystick_pos = QPointF(
                self.joystick_center.x() + dist * math.cos(angle),
                self.joystick_center.y() + dist * math.sin(angle)
            )
            self.update()

            # 计算归一化数值 (-1, 1)
            x_norm = (self.joystick_pos.x() - self.joystick_center.x()) / self.radius
            y_norm = (self.joystick_pos.y() - self.joystick_center.y()) / self.radius
            self.joystick_moved.emit(x_norm, y_norm)  # 发送信号

    def mouseReleaseEvent(self, event):
        self.dragging = False
        self.joystick_pos = self.joystick_center  # 自动回中
        self.update()
        self.joystick_moved.emit(0.0, 0.0)  # 释放时归零


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("6D Force Joystick Controller")

        # 主布局
        main_layout = QVBoxLayout()
        joystick_layout = QHBoxLayout()  # 上方两个摇杆
        bottom_layout = QHBoxLayout()  # 底部一个摇杆

        # 创建摇杆
        self.joystick_left = Joystick()  # 左摇杆控制 Fx, Fy
        self.joystick_right = Joystick()  # 右摇杆控制 Fz, Mz
        self.joystick_bottom = Joystick()  # 下摇杆控制 Mx, My

        # 添加到布局
        joystick_layout.addWidget(self.joystick_left)
        joystick_layout.addWidget(self.joystick_right)
        bottom_layout.addWidget(self.joystick_bottom)

        # 组合布局
        main_layout.addLayout(joystick_layout)
        main_layout.addLayout(bottom_layout)
        self.setLayout(main_layout)

        # 连接信号到控制方法
        self.joystick_left.joystick_moved.connect(self.update_fx_fy)
        self.joystick_right.joystick_moved.connect(self.update_fz_mz)
        self.joystick_bottom.joystick_moved.connect(self.update_mx_my)

        # 初始 6 维力
        self.force = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def update_fx_fy(self, x, y):
        """左摇杆控制 Fx, Fy"""
        self.force[0] = x * MAX_FORCE  # Fx
        self.force[1] = y * MAX_FORCE  # Fy
        self.print_force()

    def update_fz_mz(self, x, y):
        """右摇杆控制 Fz, Mz"""
        self.force[2] = y * MAX_FORCE  # Fz
        self.force[5] = x * MAX_TORQUE  # Mz
        self.print_force()

    def update_mx_my(self, x, y):
        """下摇杆控制 Mx, My"""
        self.force[3] = x * MAX_TORQUE  # Mx
        self.force[4] = y * MAX_TORQUE  # My
        self.print_force()

    def print_force(self):
        """终端打印 6 维力"""
        print(f"Force Output: Fx={self.force[0]:.2f}, Fy={self.force[1]:.2f}, Fz={self.force[2]:.2f}, "
              f"Mx={self.force[3]:.2f}, My={self.force[4]:.2f}, Mz={self.force[5]:.2f}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
