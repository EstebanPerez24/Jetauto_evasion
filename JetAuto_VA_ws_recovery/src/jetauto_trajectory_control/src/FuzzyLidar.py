#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from jetauto_interfaces.msg import imu_encoder
from std_msgs.msg import String
from geometry_msgs.msg import Pose, Point, Twist
from std_msgs.msg import Float32MultiArray
from sklearn.cluster import DBSCAN
import numpy as np
import cv2
import os
from sklearn.cluster import DBSCAN 

class Fuzzy_Laserscan:
    def __init__(self):
        rospy.init_node("Fuzzy_Lidar", anonymous=False)

        # Subscribers
        rospy.Subscriber('/jetauto_odom', Odometry, self.odom_callback)
        rospy.Subscriber('/imu_encoder', imu_encoder, self.imu_callback)
        rospy.Subscriber('/scan', LaserScan, self.lidar_callback)

        # Publishers
        self.centroids_pub = rospy.Publisher('/closest_centroids', Float32MultiArray, queue_size=10)

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

    def normalize_angle(self, angle):
        """Normaliza un ángulo entre -pi y pi."""
        return -np.arctan2(np.sin(angle), np.cos(angle))

    def lidar_callback(self, msg):
        angle_min_rad = np.pi / 2  # 90°
        angle_max_rad = 3 * np.pi / 2  # 270°

        x_points = []
        y_points = []

        for i in range(len(msg.ranges)):
            distance = msg.ranges[i]
            angle = msg.angle_min + i * msg.angle_increment
            #angle = angle % (2 * np.pi)

            if -500 <= angle <= 500:
                if msg.range_min <= distance <= min(1.5, msg.range_max) and not np.isnan(distance) and not np.isinf(distance):
                    converted_angle = np.pi - angle
                    x = distance * np.cos(converted_angle)
                    y = distance * np.sin(converted_angle)
                    x_points.append(x)
                    y_points.append(y)

        if len(x_points) >= 3:
            points = np.column_stack((x_points, y_points))
            clustering = DBSCAN(eps=0.02, min_samples=3).fit(points)
            labels = clustering.labels_

            unique_labels = set(labels)
            best_cluster = None
            min_dist = float('inf')

            for label in unique_labels:
                if label == -1:
                    continue  # ruido
                cluster_points = points[labels == label]
                cx, cy = np.mean(cluster_points, axis=0)
                dist = np.hypot(cx, cy)
                if dist < min_dist:
                    min_dist = dist
                    best_cluster = (cx, cy)

            if best_cluster:
                centroid_x, centroid_y = best_cluster
                distance_to_centroid = np.hypot(centroid_x, centroid_y)
                angle_to_centroid = self.normalize_angle(np.arctan2(centroid_y, centroid_x))

                centroid_data = [distance_to_centroid, np.degrees(angle_to_centroid)]
                rospy.loginfo(f"[CLÚSTER MÁS CERCANO] r = {distance_to_centroid:.2f}, θ = {np.degrees(angle_to_centroid):.2f}°")
            else:
                centroid_data = [1.0, 360.0]
                rospy.loginfo("No se detectó ningún clúster válido.")
        else:
            centroid_data = [1.0, 360.0]
            rospy.loginfo("No se detectaron suficientes puntos para agrupar.")

        centroids_msg = Float32MultiArray()
        centroids_msg.data = centroid_data
        self.centroids_pub.publish(centroids_msg)

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    try:
        node = Fuzzy_Laserscan()
        node.run()
    except rospy.ROSInterruptException:
        pass
