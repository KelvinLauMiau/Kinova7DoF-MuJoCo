import pinocchio as pin


class PinSolver:
    """ Pinocchio solver for kinematics and dynamics """

    def __init__(self, urdf_path: str):
        # Create data required by the algorithms
        # Load the urdf model
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        self._JOINT_NUM = self.model.nq

    def get_inertia_mat(self, q):
        """ Computing the inertia matrix in the joint frame

        :param q: joint position
        :return: inertia matrix
        """
        return pin.crba(self.model, self.data, q).copy()

    def get_coriolis_mat(self, q, qdot):
        """ Computing the Coriolis matrix in the joint frame

        :param q: joint position
        :param qdot: joint velocity
        :return:
        """
        return pin.computeCoriolisMatrix(self.model, self.data, q, qdot).copy()

    def get_gravity_mat(self, q):
        """ Computing the gravity matrix in the joint frame

        :param q: joint position
        :return: gravity matrix
        """
        return pin.computeGeneralizedGravity(self.model, self.data, q).copy()
