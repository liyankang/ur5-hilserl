"""
This file starts a control server running on the real time PC connected to the franka robot.
In a screen run `python franka_server.py`
"""
from flask import Flask, request, jsonify
import numpy as np
import rospy
import time
import subprocess
from scipy.spatial.transform import Rotation as R
from absl import app, flags

from franka_msgs.msg import ErrorRecoveryActionGoal, FrankaState
from franka_msgs.srv import SetLoad
from serl_franka_controllers.msg import ZeroJacobian
import geometry_msgs.msg as geom_msg
from dynamic_reconfigure.client import Client as ReconfClient

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "robot_ip", "172.16.0.2", "IP address of the franka robot's controller box"
)
flags.DEFINE_string(
    "gripper_ip", "192.168.1.114", "IP address of the robotiq gripper if being used"
)
flags.DEFINE_string(
    "gripper_type", "Robotiq", "Type of gripper to use: Robotiq, Franka, or None"
)
flags.DEFINE_list(
    "reset_joint_target",
    [0, 0, 0, -1.9, -0, 2, 0],
    "Target joint angles for the robot to reset to",
)
flags.DEFINE_string("flask_url", 
    "127.0.0.1",
    "URL for the flask server to run on."
)
flags.DEFINE_string("ros_port", "11311", "Port for the ROS master to run on.")


class FrankaServer:
    """
    负责管理 Franka 机器人的阻抗控制器（cartesian impedance controller）的启动与停止，
    并提供关节重置（joint reset）等恢复策略。
    
    核心功能：
      - 启动/停止笛卡尔阻抗控制器（用于柔顺运动控制）
      - 发布目标末端位姿（equilibrium pose）
      - 订阅机器人状态（位置、速度、力、雅可比等）
      - 在需要时切换到关节位置控制器进行复位
      - 清除机器人错误状态
    """

    def __init__(self, robot_ip, gripper_type, ros_pkg_name, reset_joint_target):
        """
        初始化 FrankaServer 实例。

        参数:
            robot_ip (str): Franka 控制器的 IP 地址（如 "172.16.0.2"）
            gripper_type (str): 夹爪类型（"Franka", "Robotiq", "None"）
            ros_pkg_name (str): ROS 包名（如 "serl_franka_controllers"），用于 roslaunch
            reset_joint_target (list): 关节复位目标角度 [q1, ..., q7]（单位：弧度）
        """
        self.robot_ip = robot_ip
        self.ros_pkg_name = ros_pkg_name
        self.reset_joint_target = reset_joint_target
        self.gripper_type = gripper_type

        # === 发布器（Publishers）===
        # 1. 发布末端执行器的目标位姿（给阻抗控制器使用）
        self.eepub = rospy.Publisher(
            "/cartesian_impedance_controller/equilibrium_pose",
            geom_msg.PoseStamped,
            queue_size=10,
        )

        # 2. 发布错误恢复命令（用于清除 Franka 的错误状态）
        self.resetpub = rospy.Publisher(
            "/franka_control/error_recovery/goal", 
            ErrorRecoveryActionGoal, 
            queue_size=1
        )

        # === 订阅器（Subscribers）===
        # 1. 订阅雅可比矩阵（用于计算末端速度）
        self.jacobian_sub = rospy.Subscriber(
            "/cartesian_impedance_controller/franka_jacobian",
            ZeroJacobian,
            self._set_jacobian,
        )
        time.sleep(1)  # 等待订阅建立（非严格必要，但提高稳定性）

        # 2. 订阅 Franka 全局状态（包含位姿、关节角、外力等）
        self.state_sub = rospy.Subscriber(
            "franka_state_controller/franka_states", 
            FrankaState, 
            self._set_currpos
        )

    def start_impedance(self):
        """启动笛卡尔阻抗控制器。
        
        通过 roslaunch 启动 impedance.launch 文件，加载阻抗控制节点。
        此控制器允许通过发布 equilibrium_pose 来柔顺地控制末端位姿。
        """
        self.imp = subprocess.Popen(
            [
                "roslaunch",
                self.ros_pkg_name,
                "impedance.launch",
                "robot_ip:=" + self.robot_ip,
                f"load_gripper:={'true' if self.gripper_type == 'Franka' else 'false'}",
            ],
            stdout=subprocess.PIPE,
        )
        time.sleep(3)  # 等待控制器完全启动

    def stop_impedance(self):
        """停止阻抗控制器进程。
        
        注意：这只是终止 roslaunch 进程，并不会自动切换回其他控制器。
        """
        self.imp.terminate()
        time.sleep(1)

    def clear(self):
        """清除 Franka 控制器的错误状态（如通信超时、碰撞等）。
        
        发布一个空的 ErrorRecoveryActionGoal 消息触发恢复。
        """
        msg = ErrorRecoveryActionGoal()
        self.resetpub.publish(msg)

    def reset_joint(self):
        """执行关节角度复位操作（常用于长时间运行后消除漂移或误差）。
        
        流程：
          1. 停止当前的阻抗控制器
          2. 清除可能的错误
          3. 启动专用的关节位置控制器（joint.launch），移动到预设关节目标
          4. 等待关节到位（最多30秒）
          5. 终止关节控制器
          6. 重新启动阻抗控制器
        
        注意：此过程会短暂中断对外部指令的响应。
        """
        # Step 1: 停止阻抗控制器
        try:
            self.stop_impedance()
            self.clear()
        except:
            print("impedance Not Running")
        time.sleep(3)
        self.clear()

        # Step 2: 设置目标关节角（通过 ROS 参数服务器）
        rospy.set_param("/target_joint_positions", self.reset_joint_target)

        # Step 3: 启动关节控制器
        self.joint_controller = subprocess.Popen(
            [
                "roslaunch",
                self.ros_pkg_name,
                "joint.launch",
                "robot_ip:=" + self.robot_ip,
                f"load_gripper:={'true' if self.gripper_type == 'Franka' else 'false'}",
            ],
            stdout=subprocess.PIPE,
        )
        time.sleep(1)
        print("RUNNING JOINT RESET")
        self.clear()

        # Step 4: 等待关节到达目标位置
        count = 0
        time.sleep(1)
        while not np.allclose(
            np.array(self.reset_joint_target), 
            np.array(self.q),
            atol=1e-2,
            rtol=1e-2,
        ):
            time.sleep(1)
            count += 1
            if count > 30:
                print("joint reset TIMEOUT")
                break

        # Step 5: 停止关节控制器
        print("RESET DONE")
        self.joint_controller.terminate()
        time.sleep(1)
        self.clear()
        print("KILLED JOINT RESET", self.pos)

        # Step 6: 重启阻抗控制器
        self.start_impedance()
        print("impedance STARTED")

    def move(self, pose: list):
        """发送末端目标位姿给阻抗控制器。
        
        参数:
            pose (list): 长度为7的列表 [x, y, z, qx, qy, qz, qw]
                         表示目标位置和四元数方向（世界坐标系下）
        
        注意：此操作仅在阻抗控制器运行时有效。
        """
        assert len(pose) == 7
        msg = geom_msg.PoseStamped()
        msg.header.frame_id = "0"  # Franka 的 base frame
        msg.header.stamp = rospy.Time.now()
        msg.pose.position = geom_msg.Point(pose[0], pose[1], pose[2])
        msg.pose.orientation = geom_msg.Quaternion(pose[3], pose[4], pose[5], pose[6])
        self.eepub.publish(msg)

    def _set_currpos(self, msg):
        """回调函数：处理来自 franka_state_controller 的状态消息。
        
        解析并缓存以下信息：
          - self.pos: 末端位姿 [x, y, z, qx, qy, qz, qw]
          - self.q: 当前关节角度 (7,)
          - self.dq: 当前关节速度 (7,)
          - self.force: 末端估计外力 (x, y, z)
          - self.torque: 末端估计外力矩 (x, y, z)
          - self.vel: 末端线速度+角速度 (6,)，通过 J * dq 计算
        """
        # 将 O_T_EE（列主序 4x4 齐次变换矩阵）转为 NumPy 矩阵
        tmatrix = np.array(list(msg.O_T_EE)).reshape(4, 4).T  # 转置为行主序
        r = R.from_matrix(tmatrix[:3, :3])  # 提取旋转矩阵
        pose = np.concatenate([tmatrix[:3, -1], r.as_quat()])  # [xyz + quat]
        self.pos = pose
        self.dq = np.array(list(msg.dq)).reshape((7,))
        self.q = np.array(list(msg.q)).reshape((7,))
        self.force = np.array(list(msg.K_F_ext_hat_K)[:3])
        self.torque = np.array(list(msg.K_F_ext_hat_K)[3:])
        
        # 尝试计算末端速度（需雅可比已订阅）
        try:
            self.vel = self.jacobian @ self.dq
        except AttributeError:
            self.vel = np.zeros(6)
            rospy.logwarn("Jacobian not set, end-effector velocity temporarily not available")

    def _set_jacobian(self, msg):
        """回调函数：接收并解析雅可比矩阵。
        
        雅可比矩阵用于将关节速度映射到末端笛卡尔速度。
        消息中的 zero_jacobian 是列优先（Fortran order）展开的 6x7 矩阵。
        """
        jacobian = np.array(list(msg.zero_jacobian)).reshape((6, 7), order="F")
        self.jacobian = jacobian


###############################################################################


def main(_):
    ROS_PKG_NAME = "serl_franka_controllers"

    ROBOT_IP = FLAGS.robot_ip
    GRIPPER_IP = FLAGS.gripper_ip
    GRIPPER_TYPE = FLAGS.gripper_type
    RESET_JOINT_TARGET = FLAGS.reset_joint_target

    webapp = Flask(__name__)

    try:
        roscore = subprocess.Popen(f"roscore -p {FLAGS.ros_port}", shell=True)
        time.sleep(1)
    except Exception as e:
        raise Exception("roscore not running", e)

    # Start ros node
    rospy.init_node("franka_control_api")

    if GRIPPER_TYPE == "Robotiq":
        from robot_servers.robotiq_gripper_server import RobotiqGripperServer

        gripper_server = RobotiqGripperServer(gripper_ip=GRIPPER_IP)
    elif GRIPPER_TYPE == "Franka":
        from robot_servers.franka_gripper_server import FrankaGripperServer

        gripper_server = FrankaGripperServer()
    elif GRIPPER_TYPE == "None":
        pass
    else:
        raise NotImplementedError("Gripper Type Not Implemented")

    """Starts impedance controller"""
    robot_server = FrankaServer(
        robot_ip=ROBOT_IP,
        gripper_type=GRIPPER_TYPE,
        ros_pkg_name=ROS_PKG_NAME,
        reset_joint_target=RESET_JOINT_TARGET,
    )
    robot_server.start_impedance()

    reconf_client = ReconfClient(
        "cartesian_impedance_controllerdynamic_reconfigure_compliance_param_node"
    )

    rospy.wait_for_service('/franka_control/set_load')
    set_load_service = rospy.ServiceProxy('/franka_control/set_load', SetLoad)


    # Route for Setting Load
    @webapp.route("/set_load", methods=["POST"])
    def set_load():
        data = request.json
        mass = data['mass']
        F_x_center_load = data['F_x_center_load']
        load_inertia = data['load_inertia']
        set_load_service(mass, F_x_center_load, load_inertia)
        print("Set mass to", mass)
        return "Set Load"

    # Route for Starting impedance
    @webapp.route("/startimp", methods=["POST"])
    def start_impedance():
        robot_server.clear()
        robot_server.start_impedance()
        return "Started impedance"

    # Route for Stopping impedance
    @webapp.route("/stopimp", methods=["POST"])
    def stop_impedance():
        robot_server.stop_impedance()
        return "Stopped impedance"
    
    # Route for pose in euler angles
    @webapp.route("/getpos_euler", methods=["POST"])
    def get_pose_euler():
        xyz = robot_server.pos[:3]
        r = R.from_quat(robot_server.pos[3:]).as_euler("xyz")
        return jsonify({"pose": np.concatenate([xyz, r]).tolist()})

    # Route for Getting Pose
    @webapp.route("/getpos", methods=["POST"])
    def get_pos():
        return jsonify({"pose": np.array(robot_server.pos).tolist()})

    @webapp.route("/getvel", methods=["POST"])
    def get_vel():
        return jsonify({"vel": np.array(robot_server.vel).tolist()})

    @webapp.route("/getforce", methods=["POST"])
    def get_force():
        return jsonify({"force": np.array(robot_server.force).tolist()})

    @webapp.route("/gettorque", methods=["POST"])
    def get_torque():
        return jsonify({"torque": np.array(robot_server.torque).tolist()})

    @webapp.route("/getq", methods=["POST"])
    def get_q():
        return jsonify({"q": np.array(robot_server.q).tolist()})

    @webapp.route("/getdq", methods=["POST"])
    def get_dq():
        return jsonify({"dq": np.array(robot_server.dq).tolist()})

    @webapp.route("/getjacobian", methods=["POST"])
    def get_jacobian():
        return jsonify({"jacobian": np.array(robot_server.jacobian).tolist()})

    # Route for getting gripper distance
    @webapp.route("/get_gripper", methods=["POST"])
    def get_gripper():
        return jsonify({"gripper": gripper_server.gripper_pos})

    # Route for Running Joint Reset
    @webapp.route("/jointreset", methods=["POST"])
    def joint_reset():
        robot_server.clear()
        robot_server.reset_joint()
        return "Reset Joint"

    # Route for Activating the Gripper
    @webapp.route("/activate_gripper", methods=["POST"])
    def activate_gripper():
        print("activate gripper")
        gripper_server.activate_gripper()
        return "Activated"

    # Route for Resetting the Gripper. It will reset and activate the gripper
    @webapp.route("/reset_gripper", methods=["POST"])
    def reset_gripper():
        print("reset gripper")
        gripper_server.reset_gripper()
        return "Reset"

    # Route for Opening the Gripper
    @webapp.route("/open_gripper", methods=["POST"])
    def open():
        print("open")
        gripper_server.open()
        return "Opened"

    # Route for Closing the Gripper
    @webapp.route("/close_gripper", methods=["POST"])
    def close():
        print("close")
        gripper_server.close()
        return "Closed"

    # Route for Closing the Gripper
    @webapp.route("/close_gripper_slow", methods=["POST"])
    def close_slow():
        print("close")
        gripper_server.close_slow()
        return "Closed"

    # Route for moving the gripper
    @webapp.route("/move_gripper", methods=["POST"])
    def move_gripper():
        gripper_pos = request.json
        pos = np.clip(int(gripper_pos["gripper_pos"]), 0, 255)  # 0-255
        print(f"move gripper to {pos}")
        gripper_server.move(pos)
        return "Moved Gripper"

    # Route for Clearing Errors (Communcation constraints, etc.)
    @webapp.route("/clearerr", methods=["POST"])
    def clear():
        robot_server.clear()
        return "Clear"

    # Route for Sending a pose command
    @webapp.route("/pose", methods=["POST"])
    def pose():
        pos = np.array(request.json["arr"])
        # print("Moving to", pos)
        robot_server.move(pos)
        return "Moved"

    # Route for getting all state information
    @webapp.route("/getstate", methods=["POST"])
    def get_state():
        return jsonify(
            {
                "pose": np.array(robot_server.pos).tolist(),
                "vel": np.array(robot_server.vel).tolist(),
                "force": np.array(robot_server.force).tolist(),
                "torque": np.array(robot_server.torque).tolist(),
                "q": np.array(robot_server.q).tolist(),
                "dq": np.array(robot_server.dq).tolist(),
                "jacobian": np.array(robot_server.jacobian).tolist(),
                "gripper_pos": gripper_server.gripper_pos,
            }
        )

    # Route for updating compliance parameters
    @webapp.route("/update_param", methods=["POST"])
    def update_param():
        reconf_client.update_configuration(request.json)
        return "Updated compliance parameters"

    webapp.run(host=FLAGS.flask_url)


if __name__ == "__main__":
    app.run(main)
