#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import rospy
import sys
import numpy as np
import rospkg
import cv2
import math
import matplotlib.pyplot as plt
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import PoseStamped
from jetauto_interfaces.msg import imu_encoder
from std_msgs.msg import Float32MultiArray
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState

class PoseControl:
    def __init__(self):
        rospy.init_node("trajectory_controller", anonymous=False)

        # APF and control parameters
        self.zeta = 1.1547
        self.eta = 0.3
        self.dstar = 0.6
        self.Qstar = 0.3
        self.control_scale = 0.7
        self.v_max = 0.509
        self.w_max = 5.0
        self.position_accuracy = 0.05
        self.kp_theta = 1.5

        # Robot dimensions and sampling time
        self.tm = rospy.get_param('tiempo_muestreo', '0.1')
        self.tf = rospy.get_param('tiempo_total', '60')
        self.r = rospy.get_param('r', '0.0485')
        self.lx = rospy.get_param('lx', '0.0975')
        self.ly = rospy.get_param('ly', '0.103')

        # Subscribers and Publishers
        rospy.Subscriber('/imu_encoder', imu_encoder, self.imu_callback)
        rospy.Subscriber('/jetauto_odom', Odometry, self.odom_callback)
        #rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        rospy.Subscriber('/filtered_lines', Float32MultiArray, self.hough_callback)  # Suscripción a las líneas de Hough


        self.control_publisher = rospy.Publisher("wheel_setpoint", Float32MultiArray, queue_size=10)
        #self.goal_publisher = rospy.Publisher('/goal_pose', PoseStamped, queue_size=10)
        #self.path_pub = rospy.Publisher('/evasion_path', Path, queue_size=10)
        #self.followed_path_pub = rospy.Publisher('/followed_path', Path, queue_size=10)

        # Initialize trajectory storage
        self.x_traj = []  # Store x positions
        self.y_traj = []  # Store y positions
        self.x_error = []  # Store x errors
        self.y_error = []  # Store y errors
        self.t = []  # Store timestamps

        # Rest of the initialization...
        self.start_time = rospy.Time.now().to_sec()

        # Robot state and goal parameters
        self.x = 0
        self.y = 0
        self.theta = 0.0
        self.hough_lines = []
        self.w1 = []
        self.w2 = []
        self.w3 = []
        self.w4 = []
        #self.followed_path = Path()
        #self.followed_path.header.frame_id = "odom"

        # Desired goal position
        self.goal_x = rospy.get_param("goal_x", 2.0)
        self.goal_y = rospy.get_param("goal_y", 2.0)
        self.goal_theta = rospy.get_param("goal_theta", 0.0)

    def hough_callback(self, msg):
        """
        Callback para recibir líneas detectadas desde otro nodo.
        El mensaje es un Float32MultiArray con la estructura:
        [x1, y1, x2, y2, x1', y1', x2', y2', ...]
        """
        data = msg.data
        if len(data) % 4 == 0:  # Asegurar que los datos lleguen en múltiplos de 4
            for i in range(0, len(data), 4):
                x1, y1, x2, y2 = data[i:i+4]
                self.hough_lines.append((x1, y1, x2, y2))

    def imu_callback(self, msg):
        self.theta = msg.angle
        self.w1 = msg.w1
        self.w2 = msg.w2
        self.w3 = msg.w3
        self.w4 = msg.w4

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        # Store current position in trajectory
        self.x_traj.append(self.x)
        self.y_traj.append(self.y)
        # Calculate and store errors
        self.x_error.append(self.goal_x - self.x)
        self.y_error.append(self.goal_y - self.y)
        # Store current time
        current_time = rospy.Time.now().to_sec() - self.start_time
        self.t.append(current_time)

    def calculate_attractive_force(self, x, y):
        goal_vector = np.array([self.goal_x - x, self.goal_y - y])
        distance_to_goal = np.linalg.norm(goal_vector)

        if distance_to_goal < self.dstar:
            return -self.zeta * goal_vector
        else:
            return -(self.dstar / distance_to_goal) * self.zeta * goal_vector

    def calculate_repulsive_force(self, x, y):
        repulsive_force = np.array([0.0, 0.0])
        epsilon = 1e-6

        if self.hough_lines:
            for (x1, y1, x2, y2) in self.hough_lines:
                obstacle_vector = np.array([x - x1, y - y1])
                distance_to_obstacle = np.linalg.norm(obstacle_vector)
                if distance_to_obstacle > epsilon and distance_to_obstacle < self.Qstar:
                    direction = obstacle_vector / distance_to_obstacle
                    repulsion_strength = self.eta * (1 / self.Qstar - 1 / distance_to_obstacle) * (1 / distance_to_obstacle**2)
                    repulsive_force += repulsion_strength * direction

        return repulsive_force


    def plot(self):
        # Plot X and Y errors over time
        plt.figure()
        plt.plot(self.t, self.x_error, label='X Error')
        plt.plot(self.t, self.y_error, label='Y Error')
        plt.xlabel('Time (s)')
        plt.ylabel('Error (m)')
        plt.title('Error Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot trajectory
        plt.figure()
        plt.plot(self.goal_x, self.goal_y, 'ro', label='Goal')
        plt.plot(self.x_traj, self.y_traj, label='Robot Trajectory')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('Robot Trajectory')
        plt.legend()
        plt.grid(True)
        plt.show()

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            distance_to_goal = np.linalg.norm([self.goal_x - self.x, self.goal_y - self.y])
            if distance_to_goal <= self.position_accuracy:
                rospy.loginfo("Goal reached")
                self.control_publisher.publish(Float32MultiArray(data=[0, 0, 0, 0]))
                break

            attractive_force = self.calculate_attractive_force(self.x, self.y)
            repulsive_force = self.calculate_repulsive_force(self.x, self.y)
            total_force = attractive_force + repulsive_force

            theta_ref = np.arctan2(-total_force[1], -total_force[0])
            error_theta = theta_ref - self.theta
            error_theta = (error_theta + np.pi) % (2 * np.pi) - np.pi

            acw = np.clip(self.kp_theta * error_theta, -self.w_max, self.w_max)
            v_ref = min(np.linalg.norm(-total_force), self.v_max)
            vx = v_ref * np.cos(self.theta)
            vy = v_ref * np.sin(self.theta)

            theta_1 = self.theta + np.pi / 4
            J1 = [np.sqrt(2) * np.cos(theta_1), np.sqrt(2) * np.sin(theta_1), np.sqrt(2) * np.cos(theta_1), np.sqrt(2) * np.sin(theta_1)]
            J2 = [np.sqrt(2) * np.sin(theta_1), -np.sqrt(2) * np.cos(theta_1), np.sqrt(2) * np.sin(theta_1), -np.sqrt(2) * np.cos(theta_1)]
            J3 = [-1 / (self.lx + self.ly), -1 / (self.lx + self.ly), 1 / (self.lx + self.ly), 1 / (self.lx + self.ly)]
            J_1 = np.array([J1, J2, J3])
            J = (self.r / 4) * J_1
            J_inv = np.linalg.pinv(J)

            w = np.dot(J_inv, np.array([[vx], [vy], [acw]]))

            w1 = w[0, 0]
            w2 = w[1, 0]
            w3 = w[2, 0]
            w4 = w[3, 0]

            a = 5.00

            w1 = np.clip(w1, -a, a)
            w2 = np.clip(w2, -a, a)
            w3 = np.clip(w3, -a, a)
            w4 = np.clip(w4, -a, a)

            self.control_publisher.publish(Float32MultiArray(data=[w1, w2, w3, w4]))
            rate.sleep()

        self.plot()

if __name__ == "__main__":
    try:
        node = PoseControl()
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    try:
        node = PoseControl()
        node.run()
    except rospy.ROSInterruptException:
        pass
