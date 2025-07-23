#!/usr/bin/env python3
import rospy
import math
import numpy as np
import tf.transformations as tft
from apriltag_ros.msg import AprilTagDetectionArray
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray

class AprilTagOrientationFollower:
    def __init__(self):
        rospy.init_node("apriltag_orientation_follower")

        # Parámetros físicos del robot
        self.r = 0.0485
        self.lx = 0.0975
        self.ly = 0.103

        # Ganancias de control
        self.k_r = 1.0        # Control de distancia (adelante/atrás)
        self.k_theta = 1.0    # Control de alineación (ángulo de llegada)
        self.k_yaw = 1.0      # Control de orientación global
        self.k_lateral = 1.0  # Control de centrado lateral
        self.target_distance = 0.6  # Distancia deseada al tag

        # Transformación cámara -> robot
        self.T_cam_robot = np.array([
            [0, 0, 1, 0.0959394346777203],
            [-1, 0, 0, 0.0],
            [0, -1, 0, 0.14],
            [0, 0, 0, 1]
        ])

        self.robot_pose_matrix = np.identity(4)

        # ROS interfaces
        self.pub_wheel = rospy.Publisher("/wheel_setpoint", Float32MultiArray, queue_size=10)
        rospy.Subscriber("/tag_detections", AprilTagDetectionArray, self.tag_callback)
        rospy.Subscriber("/odom", Odometry, self.odom_callback)

        rospy.loginfo("Seguidor de orientación y centrado de AprilTag activado.")
        rospy.spin()

    def pose_to_matrix(self, pose):
        trans = np.array([pose.position.x, pose.position.y, pose.position.z])
        quat = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        T = tft.quaternion_matrix(quat)
        T[0:3, 3] = trans
        return T

    def odom_callback(self, msg):
        self.robot_pose_matrix = self.pose_to_matrix(msg.pose.pose)

    def tag_callback(self, msg):
        if not msg.detections:
            self.pub_wheel.publish(Float32MultiArray(data=[0, 0, 0, 0]))
            return

        tag = msg.detections[0]
        pose_cam = tag.pose.pose.pose
        T_tag_cam = self.pose_to_matrix(pose_cam)

        # Transformar desde cámara hasta el mundo
        T_tag_robot = np.dot(self.T_cam_robot, T_tag_cam)
        T_tag_world = np.dot(self.robot_pose_matrix, T_tag_robot)

        # Posición del tag en el marco del robot
        x_tag = T_tag_robot[0, 3]
        y_tag = T_tag_robot[1, 3]
        r = math.hypot(x_tag, y_tag)
        theta = math.atan2(y_tag, x_tag)

        # Yaw del tag respecto al mundo
        quat_tag_world = tft.quaternion_from_matrix(T_tag_world)
        _, _, yaw_tag = tft.euler_from_quaternion(quat_tag_world)

        # Yaw del robot
        quat_robot = tft.quaternion_from_matrix(self.robot_pose_matrix)
        _, _, yaw_robot = tft.euler_from_quaternion(quat_robot)

        # Error de orientación (corregido para que frente del tag sea 0 rad)
        e_yaw = (yaw_tag - yaw_robot) + (1.57)
        e_yaw = math.atan2(math.sin(e_yaw), math.cos(e_yaw))

        # Controladores
        e_r = r - self.target_distance
        v = self.k_r * e_r * math.cos(theta)
        vy = self.k_lateral * y_tag  # Movimiento lateral (centrado)
        w_track = self.k_theta * theta
        w_yaw = self.k_yaw * e_yaw
        w = w_track + w_yaw

        # Jacobiano inverso para ruedas mecanum
        th1 = np.pi / 4
        r2 = np.sqrt(2)
        J_inv = np.array([
            [ r2 * math.cos(th1),  r2 * math.sin(th1), -(self.lx + self.ly)],
            [ r2 * math.sin(th1), -r2 * math.cos(th1), -(self.lx + self.ly)],
            [ r2 * math.cos(th1),  r2 * math.sin(th1),  (self.lx + self.ly)],
            [ r2 * math.sin(th1), -r2 * math.cos(th1),  (self.lx + self.ly)],
        ])
        u = np.array([[v], [vy], [w]])
        w_vector = (1 / self.r) * np.dot(J_inv, u)
        w_clipped = np.clip(w_vector.flatten(), -9.0, 9.0)

        # Debug info
        self.pub_wheel.publish(Float32MultiArray(data=w_clipped.tolist()))

if __name__ == "__main__":
    try:
        AprilTagOrientationFollower()
    except rospy.ROSInterruptException:
        pass