#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion
from jetauto_interfaces.msg import imu_encoder
from std_msgs.msg import String
from geometry_msgs.msg import Pose, Point, Twist
from std_msgs.msg import Float32MultiArray
import numpy as np
import cv2
import os
from sklearn.cluster import DBSCAN 

class PoseControl:
    def __init__(self):
        rospy.init_node("Lidar_HoughLines", anonymous=False)
        
        # Subscribers
        rospy.Subscriber('/jetauto_odom', Odometry, self.odom_callback)
        rospy.Subscriber('/imu_encoder', imu_encoder, self.imu_callback)
        rospy.Subscriber('/scan', LaserScan, self.lidar_callback)

        # Publishers
        self.lines_pub = rospy.Publisher('/filtered_lines', Float32MultiArray, queue_size=10)
        self.obstacle_info_pub = rospy.Publisher('/obstacle_dimensions', Float32MultiArray, queue_size=10)  # Nuevo publicador

        # Robot's pose
        self.x = 0
        self.y = 0
        self.theta = 0.0

        rospy.sleep(1.0)

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

    def imu_callback(self, msg):
        self.theta = msg.angle
        self.w1 = msg.w1
        self.w2 = msg.w2
        self.w3 = msg.w3
        self.w4 = msg.w4

    def lidar_callback(self, msg):
        points = []
        points_hl = []

        min_x, max_x = float("inf"), float("-inf")
        min_y, max_y = float("inf"), float("-inf")

        for i in range(-500, 500):
            distance = msg.ranges[i]
            if msg.range_min <= distance <= min(0.8, msg.range_max):
                angle = msg.angle_min + i * msg.angle_increment
                x_lidar = distance * np.cos(angle)
                y_lidar = distance * np.sin(angle)

                theta_corrected = (self.theta + np.pi) % (2 * np.pi) - np.pi

                x_world = self.x + np.sqrt(x_lidar ** 2 + y_lidar ** 2) * np.cos(theta_corrected + angle + np.pi)
                y_world = self.y + np.sqrt(x_lidar ** 2 + y_lidar ** 2) * np.sin(theta_corrected + angle + np.pi)

                points.append((x_world, y_world))

                # Calcular dimensiones del obstáculo
                min_x = min(min_x, x_world)
                max_x = max(max_x, x_world)
                min_y = min(min_y, y_world)
                max_y = max(max_y, y_world)

                points_hl.append((int(x_world * 100 + 250), int(y_world * 100 + 250)))

        img = np.zeros((1000, 1000), dtype=np.uint8)
        for point in points_hl:
            cv2.circle(img, point, 1, 255, -1)
            #self.save_lidar_pixels(points_hl)


        lines = self.detect_lines(img)

        if lines is not None:
            filtered_lines = self.filter_redundant_lines(lines)
            self.publish_filtered_lines(filtered_lines)
            self.save_lines_to_file(filtered_lines)  # ← Guardar aquí
            for (x1, y1, x2, y2) in filtered_lines:
                cv2.line(img, (int(x1 * 100 + 250), int(y1 * 100 + 250)), 
                         (int(x2 * 100 + 250), int(y2 * 100 + 250)), (200, 200, 200), 2)

        cv2.imshow("Lidar Hough Lines", img)
        cv2.waitKey(1)

        # Detectar múltiples obstáculos usando clustering
        self.detect_obstacles(points)

    def detect_obstacles(self, points):
        if len(points) == 0:
            return

        points = np.array(points)
        
        # Aplicar DBSCAN para agrupar los puntos en diferentes obstáculos
        clustering = DBSCAN(eps=0.2, min_samples=3).fit(points)

        unique_labels = set(clustering.labels_)
        obstacle_data = []

        for label in unique_labels:
            if label == -1:
                continue  # Ignorar puntos aislados

            # Extraer los puntos de este obstáculo
            cluster_points = points[clustering.labels_ == label]
            min_x, max_x = np.min(cluster_points[:, 0]), np.max(cluster_points[:, 0])
            min_y, max_y = np.min(cluster_points[:, 1]), np.max(cluster_points[:, 1])

            width = max_x - min_x
            height = max_y - min_y

            # Agregar datos al mensaje a publicar
            obstacle_data.extend([min_x, max_x, min_y, max_y, width, height])

        # Publicar información de los obstáculos detectados
        msg = Float32MultiArray(data=obstacle_data)
        self.obstacle_info_pub.publish(msg)
        
    def detect_lines(self, img):
        lines_detected = cv2.HoughLinesP(img, 1, np.pi / 180, threshold=5, minLineLength=5, maxLineGap=5)
        lines = []
        if lines_detected is not None:
            for line in lines_detected:
                x1, y1, x2, y2 = line[0]
                x1_world = (x1 - 250) / 100.0
                y1_world = (y1 - 250) / 100.0
                x2_world = (x2 - 250) / 100.0
                y2_world = (y2 - 250) / 100.0
                lines.append((x1_world, y1_world, x2_world, y2_world))
        return lines

    def filter_redundant_lines(self, lines, distance_threshold=0.1, angle_threshold=5):
        unique_lines = []

        def line_distance(l1, l2):
            return np.sqrt((l1[0] - l2[0]) ** 2 + (l1[1] - l2[1]) ** 2 +
                           (l1[2] - l2[2]) ** 2 + (l1[3] - l2[3]) ** 2)

        def line_angle(l1, l2):
            angle1 = np.arctan2(l1[3] - l1[1], l1[2] - l1[0])
            angle2 = np.arctan2(l2[3] - l2[1], l2[2] - l2[0])
            return np.abs(np.degrees(angle1 - angle2))

        for line in lines:
            is_unique = True
            for unique_line in unique_lines:
                if (line_distance(line, unique_line) < distance_threshold and
                        line_angle(line, unique_line) < angle_threshold):
                    is_unique = False
                    break
            if is_unique:
                unique_lines.append(line)

        return unique_lines

    def publish_filtered_lines(self, lines):
        msg = Float32MultiArray()
        for line in lines:
            msg.data.extend(line)
        self.lines_pub.publish(msg)

    def save_lines_to_file(self, lines, filename="hough_lines.txt"):
        save_path = "/home/jetauto/JetAuto_VA_ws/src/jetauto_trajectory_control/Lidar/hough_lines_dinam.txt"
        try:
            f = open(save_path, "a")
            for line in lines:
                f.write("%.3f, %.3f, %.3f, %.3f\n" % (line[0], line[1], line[2], line[3]))
            f.close()
        except Exception as e:
            rospy.logerr("Error al guardar líneas Hough: %s" % str(e))

    def save_lidar_pixels(self, pixel_points, filename="lidar_pixels.txt"):
        save_path = "/home/jetauto/JetAuto_VA_ws/src/jetauto_trajectory_control/Lidar/lidar_pixels_caja.txt"
        try:
            f = open(save_path, "a")
            for point in pixel_points:
                f.write("%d, %d\n" % (point[0], point[1]))
            f.close()
        except Exception as e:
            rospy.logerr("Error al guardar puntos LIDAR en píxeles: %s" % str(e))




    def run(self):
        rospy.spin()

if __name__ == "__main__":
    try:
        node = PoseControl()
        node.run()
    except rospy.ROSInterruptException:
        pass
