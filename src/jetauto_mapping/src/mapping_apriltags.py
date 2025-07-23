#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import math
from shapely.geometry import Point, Polygon
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Pose
from apriltag_ros.msg import AprilTagDetectionArray
from tf.transformations import quaternion_matrix, euler_from_quaternion, quaternion_from_matrix

# Variables globales
mapped_tags = {}
half_w = 0.15
half_h = 0.01
robot_pose_matrix = np.identity(4)
current_odom = None
xmax = 1.5
ymax = 1.5
xmin = -1.5
ymin = -1.5

# --------------------------------------
def pose_to_homogeneous(pose):
    trans = np.array([pose.position.x, pose.position.y, pose.position.z])
    quat = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
    T = quaternion_matrix(quat)
    T[0:3, 3] = trans
    return T

def create_obstacle_polygon(x, y, theta):
    corners_local = [(-half_w, -half_h), (+half_w, -half_h), (+half_w, +half_h), (-half_w, +half_h)]
    corners_global = []
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    for (lx, ly) in corners_local:
        gx = lx * cos_t - ly * sin_t
        gy = lx * sin_t + ly * cos_t
        gx += x
        gy += y
        corners_global.append((gx, gy))
    return Polygon(corners_global)

def place_obstacle_in_map(occ_grid, obstacle_poly):
    resolution = occ_grid.info.resolution
    width = occ_grid.info.width
    height = occ_grid.info.height
    origin_x = occ_grid.info.origin.position.x
    origin_y = occ_grid.info.origin.position.y
    minx, miny, maxx, maxy = obstacle_poly.bounds
    min_col = max(0, int((minx - origin_x) / resolution))
    max_col = min(width - 1, int((maxx - origin_x) / resolution))
    min_row = max(0, int((miny - origin_y) / resolution))
    max_row = min(height - 1, int((maxy - origin_y) / resolution))
    data = list(occ_grid.data)
    for col in range(min_col, max_col + 1):
        for row in range(min_row, max_row + 1):
            idx = row * width + col
            center_x = origin_x + (col + 0.5) * resolution
            center_y = origin_y + (row + 0.5) * resolution
            pt = Point(center_x, center_y)
            if obstacle_poly.contains(pt):
                data[idx] = 100
    occ_grid.data = data

def odom_callback(msg):
    global robot_pose_matrix, current_odom
    pose = msg.pose.pose
    current_odom = pose
    robot_pose_matrix = pose_to_homogeneous(pose)

def callback_tag_detections(msg):
    global occ_grid

    for detection in msg.detections:
        tag_id = detection.id[0]

        if tag_id in mapped_tags:
            continue  # Ya fue mapeado

        pose_cam = detection.pose.pose.pose
        dx = pose_cam.position.x
        dy = pose_cam.position.y
        dz = pose_cam.position.z
        distance_to_tag = math.sqrt(dx**2 + dy**2 + dz**2)

        if distance_to_tag > 2.0:
            continue  # Muy lejos, ignorar

        # Transformar a frame del mundo
        T_tag_cam = pose_to_homogeneous(pose_cam)

        T_cam_robot = np.array([
            [0, 0, 1, 0.0959394346777203],
            [-1, 0, 0, 0.0],
            [0, -1, 0, 0.14],
            [0, 0, 0, 1]
        ])

        T_tag_robot = np.matmul(T_cam_robot, T_tag_cam)
        T_tag_world = np.matmul(robot_pose_matrix, T_tag_robot)

        x = T_tag_world[0, 3]
        y = T_tag_world[1, 3]
        quat = quaternion_from_matrix(T_tag_world)
        _, _, yaw = euler_from_quaternion(quat)
        yaw_deg = math.degrees(yaw)
        if yaw_deg < 0:
            yaw_deg += 360.0

        # Crear obstáculo y marcar en el mapa
        poly = create_obstacle_polygon(x, y, yaw)
        place_obstacle_in_map(occ_grid, poly)

        mapped_tags[tag_id] = (x, y)
        occ_grid.header.stamp = rospy.Time.now()
        map_pub.publish(occ_grid)

        rospy.loginfo("[Tag {}] MAPEADO en (x={:.2f}, y={:.2f}), yaw={:.1f}°".format(tag_id, x, y, yaw_deg))

# --------------------------------------

if __name__ == "__main__":
    rospy.init_node("apriltag_map_simple")

    half_w = rospy.get_param("~half_width", 0.15)
    half_h = rospy.get_param("~half_height", 0.05)
    resolution = rospy.get_param("~resolution", 0.01)
    xmin = rospy.get_param("~xmin", -1.5)
    xmax = rospy.get_param("~xmax", 1.5)
    ymin = rospy.get_param("~ymin", -1.5)
    ymax = rospy.get_param("~ymax", 1.5)

    width = int((xmax - xmin) / resolution)
    height = int((ymax - ymin) / resolution)

    occ_grid = OccupancyGrid()
    occ_grid.header.frame_id = "odom"
    occ_grid.info.resolution = resolution
    occ_grid.info.width = width
    occ_grid.info.height = height
    occ_grid.info.origin.position.x = xmin
    occ_grid.info.origin.position.y = ymin
    occ_grid.info.origin.orientation.w = 1.0
    occ_grid.data = [0] * (width * height)

    map_pub = rospy.Publisher("/map", OccupancyGrid, latch=True, queue_size=1)
    rospy.Subscriber("/tag_detections", AprilTagDetectionArray, callback_tag_detections)
    rospy.Subscriber("/jetauto_odom", Odometry, odom_callback)

    rospy.spin()


