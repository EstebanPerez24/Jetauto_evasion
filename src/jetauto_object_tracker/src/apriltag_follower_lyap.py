#!/usr/bin/env python3
import rospy
import math
import numpy as np
import tf.transformations as tft
from apriltag_ros.msg import AprilTagDetectionArray
from std_msgs.msg import Float32MultiArray

class AprilTagWheelFollower:
    def __init__(self):
        rospy.init_node("apriltag_follower_wheel_control")

        # Parámetros del robot
        self.r = 0.0485
        self.lx = 0.0975
        self.ly = 0.103

        # Controlador Lyapunov
        self.target_distance = 0.6
        self.k_r = 1.0
        self.k_theta = 3.0

        # Publicador de velocidades de rueda
        self.wheel_pub = rospy.Publisher("/wheel_setpoint", Float32MultiArray, queue_size=10)

        # Transformación fija de la cámara respecto al robot
        self.T_cam_robot = np.array([
            [0, 0, 1, 0.0959394346777203],
            [-1, 0, 0, 0.0],
            [0, -1, 0, 0.14],
            [0, 0, 0, 1]
        ])

        rospy.Subscriber("/tag_detections", AprilTagDetectionArray, self.detection_callback)
        rospy.loginfo("Controlador Lyapunov")
        rospy.spin()

    def detection_callback(self, msg):
        if not msg.detections:
            self.wheel_pub.publish(Float32MultiArray(data=[0, 0, 0, 0]))
            return

        tag = msg.detections[0]
        p = tag.pose.pose.pose.position
        q = tag.pose.pose.pose.orientation

        # Transformar pose del tag desde cámara al robot
        T_tag_cam = tft.quaternion_matrix([q.x, q.y, q.z, q.w])
        T_tag_cam[0, 3] = p.x
        T_tag_cam[1, 3] = p.y
        T_tag_cam[2, 3] = p.z
        T_tag_robot = np.dot(self.T_cam_robot, T_tag_cam)

        x_tag = T_tag_robot[0, 3]
        y_tag = T_tag_robot[1, 3]
        r = math.hypot(x_tag, y_tag)
        theta = math.atan2(y_tag, x_tag)

        # Lyapunov: velocidad de avance y rotación
        e_r = r - self.target_distance
        v = self.k_r * e_r * math.cos(theta)
        w = self.k_theta * theta

        # Vector de velocidades: avance, lateral (0), rotación
        u = np.array([[v], [0], [w]])

        # Jacobiano inverso mecanum
        th1 = np.pi / 4  # θ = 0 (robot alineado con tag en su frame)
        r2 = np.sqrt(2)
        J_inv = np.array([
            [ r2 * math.cos(th1),  r2 * math.sin(th1), -(self.lx + self.ly)],
            [ r2 * math.sin(th1), -r2 * math.cos(th1), -(self.lx + self.ly)],
            [ r2 * math.cos(th1),  r2 * math.sin(th1),  (self.lx + self.ly)],
            [ r2 * math.sin(th1), -r2 * math.cos(th1),  (self.lx + self.ly)],
        ])

        w_vector = (1 / self.r) * np.dot(J_inv, u)
        w1, w2, w3, w4 = w_vector.flatten()

        # Saturación opcional
        max_w = 9.0
        w1 = max(min(w1, max_w), -max_w)
        w2 = max(min(w2, max_w), -max_w)
        w3 = max(min(w3, max_w), -max_w)
        w4 = max(min(w4, max_w), -max_w)

        self.wheel_pub.publish(Float32MultiArray(data=[w1, w2, w3, w4]))

if __name__ == "__main__":
    try:
        AprilTagWheelFollower()
    except rospy.ROSInterruptException:
        pass