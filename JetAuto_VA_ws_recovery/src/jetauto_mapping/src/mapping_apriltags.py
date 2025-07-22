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
half_w = 0.1
half_h = 0.01
robot_pose_matrix = np.identity(4)
tag_data_buffer = {}
seen_now = set()
current_odom = None

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

def ransac_pose_estimate_2d(poses_world, threshold=0.1, max_iterations=100):
    best_inliers = []
    best_model = None
    data = np.array(poses_world)

    for _ in range(max_iterations):
        idx = np.random.choice(len(data), 1)
        sample = data[idx][0]
        inliers = []

        for point in data:
            dist = np.linalg.norm(point[:2] - sample[:2])
            if dist < threshold:
                inliers.append(point)

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_model = sample

    if not best_inliers:
        return None

    inliers_np = np.array(best_inliers)
    x_avg = np.mean(inliers_np[:, 0])
    y_avg = np.mean(inliers_np[:, 1])
    yaw_sin = np.mean(np.sin(inliers_np[:, 2]))
    yaw_cos = np.mean(np.cos(inliers_np[:, 2]))
    yaw_avg = math.atan2(yaw_sin, yaw_cos)

    return x_avg, y_avg, yaw_avg

def odom_callback(msg):
    global robot_pose_matrix, current_odom
    pose = msg.pose.pose
    current_odom = pose
    robot_pose_matrix = pose_to_homogeneous(pose)

def callback_tag_detections(msg):
    seen_now.clear()
    for detection in msg.detections:
        tag_id = detection.id[0]
        seen_now.add(tag_id)

        if tag_id in mapped_tags:
            continue

        pose_cam = detection.pose.pose.pose
        dx = pose_cam.position.x
        dy = pose_cam.position.y
        dz = pose_cam.position.z
        distance_to_tag = math.sqrt(dx**2 + dy**2 + dz**2)
        if distance_to_tag > 1.0:
            continue

        if tag_id not in tag_data_buffer:
            tag_data_buffer[tag_id] = {
                'detections': [],
                'current_series': [],
                'seen': False,
                'count': 0,
                'mapped': False
            }

        entry = tag_data_buffer[tag_id]

        if not entry['seen']:
            entry['current_series'] = []
            entry['seen'] = True

        entry['current_series'].append(pose_cam)

        if len(entry['current_series']) >= 10:
            entry['detections'].append(entry['current_series'])
            entry['count'] += 1
            entry['seen'] = False
            rospy.loginfo("[Tag {}] Aparición {} registrada.".format(tag_id, entry['count']))

    for tag_id, entry in tag_data_buffer.items():
        if entry['mapped'] or tag_id not in seen_now:
            continue

        if entry['count'] >= 3:
            all_poses = sum(entry['detections'], [])
            poses_world = []

            for p in all_poses:
                T_tag_cam = pose_to_homogeneous(p)
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

                poses_world.append((x, y, yaw))

            result = ransac_pose_estimate_2d(poses_world)
            if result is None:
                rospy.logwarn("RANSAC falló para el tag {}".format(tag_id))
                continue

            x_tag_world, y_tag_world, yaw_rad = result
            yaw_deg = math.degrees(yaw_rad)
            if yaw_deg < 0:
                yaw_deg += 360.0

            poly = create_obstacle_polygon(x_tag_world, y_tag_world, yaw_rad)
            place_obstacle_in_map(occ_grid, poly)
            mapped_tags[tag_id] = (x_tag_world, y_tag_world)
            entry['mapped'] = True
            occ_grid.header.stamp = rospy.Time.now()
            map_pub.publish(occ_grid)
            rospy.loginfo("[Tag {}] MAPEADO en (x={:.2f}, y={:.2f}), yaw={:.1f}°".format(tag_id, x_tag_world, y_tag_world, yaw_deg))

# --------------------------------------

if __name__ == "__main__":
    rospy.init_node("apriltag_map_navigator_ransac")

    half_w = rospy.get_param("~half_width", 0.1)
    half_h = rospy.get_param("~half_height", 0.05)
    resolution = rospy.get_param("~resolution", 0.01)
    xmin = rospy.get_param("~xmin", -5.0)
    xmax = rospy.get_param("~xmax", 5.0)
    ymin = rospy.get_param("~ymin", -5.0)
    ymax = rospy.get_param("~ymax", 5.0)

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

