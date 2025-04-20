import mujoco
from mujoco import viewer
import numpy as np
from joint_impedance_controller import JntImpedance  # 假设控制器已经定义好

class Robot:
    ACTUATORS = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6']  # 6自由度

    def __init__(self, control_freq=50, target_angles=None) -> None:
        if target_angles is None:
            target_angles = np.zeros(7)  # 默认目标关节位置是零位置
        self.target_angles = target_angles  # 用户输入的目标关节角度

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
        """ 渲染一帧图像 """
        if self.viewer.is_running():
            self.viewer.sync()

    def step(self, action: np.ndarray):
        """ 执行一步仿真 """
        for i in range(self._n_substeps):
            self.inner_step(action)
            mujoco.mj_step(self.mj_model, self.mj_data)

    def inner_step(self, action):
        # 计算阻抗控制器的目标力矩
        torque = self.controller.compute_jnt_torque(
            q_des=self.target_angles,  # 目标关节位置
            v_des=np.zeros(7),         # 目标速度为零
            q_cur=self.mj_data.qpos,   # 当前关节位置
            v_cur=self.mj_data.qvel,   # 当前关节速度
        )

        # 将力矩发送到每个关节
        for j, per_actuator_index in enumerate(self.ACTUATORS):
            self.mj_data.actuator(per_actuator_index).ctrl = torque[j]

    def update_target_angles(self, new_target_angles):
        """ 更新目标关节角度 """
        self.target_angles = new_target_angles

if __name__ == '__main__':
    # 创建一个机器人实例，目标关节角度为一个输入数组
    target_angles = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])  # 用户输入的目标关节角度
    robot = Robot(target_angles=target_angles)

    # 在此进行动态输入（可以通过外部接口接收实时控制命令）
    for _ in range(int(1e5)):
        robot.step(robot.mj_data.qpos)  # 通过当前关节位置进行一步仿真
        robot.render()  # 渲染当前帧

        # 假设用户改变了目标关节位置
        # 可以通过更新目标角度来模拟不同的行为
        # new_target_angles =  np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 新的目标关节角度
        # robot.update_target_angles(new_target_angles)

