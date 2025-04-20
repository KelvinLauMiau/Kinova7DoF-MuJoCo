import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QVector3D
from PyQt6.QtCore import Qt
import numpy as np
import sys

class CubeController(gl.GLViewWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Cube Joystick")
        self.setGeometry(100, 100, 600, 600)
        self.show()

        # ✅ 创建 3D 立方体
        self.cube = gl.GLBoxItem(size=QVector3D(1, 1, 1))
        self.addItem(self.cube)

        # 初始 6 维力
        self.force = np.zeros(6)

        # 记录鼠标状态
        self.last_mouse_pos = None

    def mouseMoveEvent(self, event):
        """ 监听鼠标拖动事件，计算 6 维力 """
        if self.last_mouse_pos is None:
            self.last_mouse_pos = event.position()
            return

        # 计算鼠标移动距离
        dx = event.position().x() - self.last_mouse_pos.x()
        dy = event.position().y() - self.last_mouse_pos.y()
        self.last_mouse_pos = event.position()

        # ✅ 计算力 Fx, Fy
        self.force[0] += dx * 0.01  # Fx (缩小步长)
        self.force[1] -= dy * 0.01  # Fy (缩小步长)

        # ✅ 计算力矩 Mx, My
        self.force[3] += dy * 0.05  # 绕 x 轴力矩
        self.force[4] += dx * 0.05  # 绕 y 轴力矩

        print(f"Force Output: Fx={self.force[0]:.2f}, Fy={self.force[1]:.2f}, Fz={self.force[2]:.2f}, "
              f"Mx={self.force[3]:.2f}, My={self.force[4]:.2f}, Mz={self.force[5]:.2f}")

    def mousePressEvent(self, event):
        """ 记录鼠标按下时的位置 """
        self.last_mouse_pos = event.position()

    def mouseReleaseEvent(self, event):
        """ 释放鼠标时清空记录 """
        self.last_mouse_pos = None

if __name__ == "__main__":
    app = QApplication(sys.argv)
    cube = CubeController()
    sys.exit(app.exec())
