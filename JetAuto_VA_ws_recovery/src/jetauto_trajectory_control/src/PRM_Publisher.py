#!/usr/bin/env python

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Path
from std_msgs.msg import Float32MultiArray, Header, Bool
from geometry_msgs.msg import PoseStamped
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, Point, Twist
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from tf.transformations import euler_from_quaternion
import numpy as np
import cv2
import os
import heapq 

class DynamicPRM:
    def __init__(self):
        rospy.init_node("dynamic_prm", anonymous=False)

        # Parameters
        self.num_nodes = rospy.get_param("~num_nodes", 60)
        self.connection_distance = rospy.get_param("~connection_distance", 2.5)
        self.threshold = rospy.get_param("~threshold", 0.2)
        self.workspace_limits = np.array([[0, 4], [-4, 4]])
        self.goal_point = np.array([2.0, -2.0])
        self.nodes = []
        self.adj_matrix = None
        self.current_path = []  # Store the current valid path
        self.initial_path_published = False  # Flag to prevent duplicate publishing

        self.x = 0
        self.y = 0
        self.theta = 0.0
        
        # Subscribers
        rospy.Subscriber("/jetauto_odom", Odometry, self.odom_callback)
        rospy.Subscriber("/filtered_lines", Float32MultiArray, self.filtered_lines_callback)


        self.collision_pub = rospy.Publisher("/collision_detected", Bool, queue_size=10)

        # Publisher for the trajectory as a Path    
        self.points_pub = rospy.Publisher("/prm_points", Float32MultiArray, queue_size=10)  

        # Initialize the PRM graph and publish the initial path
        self.initialize_prm()

    def initialize_prm(self):
        if self.x is None or self.y is None:
            rospy.logwarn("Robot pose not initialized. Waiting for odometry data...")
            rospy.Timer(rospy.Duration(1), lambda _: self.initialize_prm(), oneshot=True)
            return

        rospy.loginfo("Generating initial PRM graph and path...")
        hough_lines = []  # No obstacles for the initial path
        self.nodes = self.generate_nodes(hough_lines)
        self.adj_matrix = self.generate_adj_matrix(self.nodes, hough_lines)
        path = self.plan_path()

        if not path:
            rospy.logwarn("Failed to generate a valid initial PRM path. Retrying...")
            rospy.Timer(rospy.Duration(1), lambda _: self.initialize_prm(), oneshot=True)
            return

        self.current_path = path
        rospy.loginfo("Initial PRM path generated successfully.")

        if not self.initial_path_published:
            rospy.sleep(1.0)  # Allow RViz to initialize
            self.publish_trajectory(self.current_path)  # Publish the initial trajectory
            self.initial_path_published = True

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        rot_q = msg.pose.pose.orientation
        (roll, pitch, self.theta) = euler_from_quaternion([rot_q.x, rot_q.y, rot_q.z, rot_q.w])

    def filtered_lines_callback(self, msg):
        # msg.data contains the flattened array of line coordinates
        lines = []
        data = msg.data
        for i in range(0, len(data), 4):
            x1, y1, x2, y2 = data[i:i+4]
            lines.append((x1, y1, x2, y2))
        
        # Log the lines for debugging
        rospy.loginfo("ReceiveD filtered lines:")

        
        self.update_prm(lines)

    def update_prm(self, hough_lines):
        if self.current_path and not self.is_path_colliding(self.current_path, hough_lines):
            rospy.loginfo("Current path is valid. No update needed.")
            return  # Path is valid; no update required

        rospy.loginfo("Collision detected with current path. Updating PRM and trajectory...")

        # Publish the collision message
        collision_msg = Bool(data=True)
        self.collision_pub.publish(collision_msg)

        self.nodes = self.generate_nodes(hough_lines)
        self.adj_matrix = self.generate_adj_matrix(self.nodes, hough_lines)
        path = self.plan_path()

        if not path:
            rospy.logwarn("Failed to generate a valid updated path. Keeping the old path.")
            return

        self.current_path = path
        rospy.loginfo("Path successfully updated.")
        self.publish_trajectory(self.current_path)


    def is_path_colliding(self, path, hough_lines):
        """
        Checks if the current path intersects with any detected Hough lines.
        :param path: List of node indices representing the current path.
        :param hough_lines: List of detected obstacle lines.
        :return: True if a collision is detected, False otherwise.
        """
        for i in range(len(path) - 1):
            p1 = self.nodes[path[i]]
            p2 = self.nodes[path[i + 1]]
            if self.check_edge_collision(p1, p2, hough_lines):
                return True
        return False

    def generate_nodes(self, hough_lines):
        if self.x is None or self.y is None:
            rospy.logwarn("Robot pose not initialized. Cannot generate nodes.")
            return []

        nodes = []
        while len(nodes) < self.num_nodes:
            candidate_node = np.random.uniform(self.workspace_limits[:, 0], self.workspace_limits[:, 1])
            if not self.is_collision(candidate_node, hough_lines):
                nodes.append(candidate_node)

        # Add the robot's current position as the start point
        start_point = np.array([self.x, self.y])
        nodes.append(start_point)

        # Add the goal point
        nodes.append(self.goal_point)
        return np.array(nodes)

    def generate_adj_matrix(self, nodes, hough_lines):
        num_nodes = len(nodes)
        adj_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                distance = np.linalg.norm(nodes[i] - nodes[j])
                if distance <= self.connection_distance and not self.check_edge_collision(nodes[i], nodes[j], hough_lines):
                    adj_matrix[i, j] = distance
                    adj_matrix[j, i] = distance
        return adj_matrix

    def plan_path(self):
        if self.x is None or self.y is None:
            rospy.logwarn("Robot pose not initialized. Cannot plan path.")
            return []

        start_idx = len(self.nodes) - 2  # The second-to-last node is the robot's current position
        goal_idx = len(self.nodes) - 1  # The last node is the goal point
        path = self.dijkstra(self.adj_matrix, start_idx, goal_idx)

        if not path or len(path) < 2:
            rospy.logwarn("Dijkstra failed to find a valid path.")
            return []
        return path


    def is_collision(self, point, lines):
        for line in lines:
            x1, y1, x2, y2 = line
            if self.point_to_segment_distance(point, np.array([x1, y1]), np.array([x2, y2])) <= self.threshold:
                return True
        return False

    def check_edge_collision(self, p1, p2, lines):
        num_steps = 100
        for t in np.linspace(0, 1, num_steps):
            intermediate_point = (1 - t) * p1 + t * p2
            if self.is_collision(intermediate_point, lines):
                return True
        return False

    def point_to_segment_distance(self, point, p1, p2):
        """
        Calculate the shortest distance between a point and a line segment.
        :param point: The point (x, y) as a NumPy array.
        :param p1: One endpoint of the segment (x, y) as a NumPy array.
        :param p2: The other endpoint of the segment (x, y) as a NumPy array.
        :return: The shortest distance between the point and the segment.
        """
        v = p2 - p1
        w = point - p1
        c1 = np.dot(w, v)
        if c1 <= 0:
            return np.linalg.norm(point - p1)
        c2 = np.dot(v, v)
        if c2 <= c1:
            return np.linalg.norm(point - p2)
        b = c1 / c2
        pb = p1 + b * v
        return np.linalg.norm(point - pb)

    def dijkstra(self, graph, start, goal):
        num_nodes = graph.shape[0]
        distances = np.full(num_nodes, np.inf)
        distances[start] = 0
        prev = [None] * num_nodes
        priority_queue = [(0, start)]

        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)
            if current_node == goal:
                break
            for neighbor in range(num_nodes):
                if graph[current_node, neighbor] > 0:
                    alt = current_distance + graph[current_node, neighbor]
                    if alt < distances[neighbor]:
                        distances[neighbor] = alt
                        prev[neighbor] = current_node
                        heapq.heappush(priority_queue, (alt, neighbor))

        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = prev[current]
        return path[::-1]

    def publish_trajectory(self, path):
        if not path or len(path) < 2:
            rospy.logwarn("No valid path to publish! Skipping trajectory generation.")
            return

        rospy.loginfo("Publishing trajectory points...")
        trajectory_points = [self.nodes[idx] for idx in path]

        # Publish points as Float32MultiArray
        points_msg = Float32MultiArray()
        points_msg.data = [coord for point in trajectory_points for coord in point]
        self.points_pub.publish(points_msg)

        # Publish the collision message
        collision_msg = Bool(data=False)
        self.collision_pub.publish(collision_msg)
        rospy.loginfo("Trajectory points published to /prm_points.")

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        node = DynamicPRM()
        node.run()
    except rospy.ROSInterruptException:
        pass
