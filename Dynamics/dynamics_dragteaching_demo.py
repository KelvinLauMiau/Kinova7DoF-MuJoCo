import mujoco
from mujoco import viewer
import numpy as np

from joint_impedance_controller import JntImpedance


class Robot:
    ACTUATORS = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7']

    def __init__(self, control_freq=20) -> None:

        self.mj_model = mujoco.MjModel.from_xml_path(filename='../Model/ActualArm/Kinova_mjmodel.xml')
        self.mj_data = mujoco.MjData(self.mj_model)

        self.viewer = viewer.launch_passive(self.mj_model, self.mj_data)

        # 计算控制频率
        control_timestep = 1.0 / control_freq
        model_timestep = self.mj_model.opt.timestep
        self._n_substeps = int(control_timestep / model_timestep)

        # 初始化控制器
        self.controller = JntImpedance(urdf_path='../Kinova_description/urdf/Kinova_description.urdf')

    def render(self):
        """ 渲染一帧图像
        """
        if self.viewer.is_running():
            self.viewer.sync()

    def step(self, action: np.ndarray):
        """
        This method will be called with one-step in mujoco
        :param action: Input action
        :return: None
        """
        for i in range(self._n_substeps):
            self.inner_step(action)
            mujoco.mj_step(self.mj_model, self.mj_data)

    def inner_step(self, action):
        torque = self.controller.compute_jnt_torque(
            q_des=action,
            v_des=np.zeros(7),
            q_cur=self.mj_data.qpos,
            v_cur=self.mj_data.qvel,
        )
        # Send joint torque to simulation
        for j, per_actuator_index in enumerate(self.ACTUATORS):
            self.mj_data.actuator(per_actuator_index).ctrl = torque[j]


if __name__ == '__main__':
    robot = Robot()
    for _ in range(int(1e5)):
        robot.step(robot.mj_data.qpos)
        robot.render()
