#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import rospy
import rospkg
import Paths as paths
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Point, Twist
from math import sqrt
import time
from std_msgs.msg import Float32MultiArray, Header
from geometry_msgs.msg import PoseStamped
from jetauto_interfaces.msg import imu_encoder
import fuzzylite as fl
import math
import numpy as np
from math import atan2
import json



class PoseControl:
    def __init__(self):
        rospy.init_node("trajectory_controller", anonymous=False)

        self.engine = self.build_fuzzy_engine("/home/jetauto/JetAuto_VA_ws/src/jetauto_trajectory_control/src/simple_control.json")

        self.r = rospy.get_param('r', 0.0485)
        self.lx = rospy.get_param('lx', 0.0975)
        self.ly = rospy.get_param('ly', 0.103)

        rospy.Subscriber('/imu_encoder', imu_encoder, self.imu_callback)
        rospy.Subscriber('/jetauto_odom', Odometry, self.odom_callback)
        
        # Target point
        self.target_x = rospy.get_param("target_x", -1.0)
        self.target_y = rospy.get_param("target_y", -1.0)
        self.max_wheel_speed = 5.0  # Saturation limit in rad/s

        self.control_publisher = rospy.Publisher("wheel_setpoint", Float32MultiArray, queue_size=10)
      
        
        self.control_publisher.publish(Float32MultiArray(data=[0, 0, 0, 0]))
        
             
        self.x = 0
        self.y = 0
        self.theta = 0.0
        self.w1 = 0.0
        self.w2 = 0.0
        self.w3 = 0.0
        self.w4 = 0.0
       
                     
    def saturate_wheel_speeds(self, speeds):
        return np.clip(speeds, -self.max_wheel_speed, self.max_wheel_speed)

    def calculate_jacobian(self):
        theta_1 = self.theta + np.pi / 4
        J1 = [np.sqrt(2) * np.cos(theta_1), np.sqrt(2) * np.sin(theta_1), np.sqrt(2) * np.cos(theta_1), np.sqrt(2) * np.sin(theta_1)]
        J2 = [np.sqrt(2) * np.sin(theta_1), -np.sqrt(2) * np.cos(theta_1), np.sqrt(2) * np.sin(theta_1), -np.sqrt(2) * np.cos(theta_1)]
        J3 = [-1 / (self.lx + self.ly), -1 / (self.lx + self.ly), 1 / (self.lx + self.ly), 1 / (self.lx + self.ly)]
        J = (self.r / 4) * np.array([J1, J2, J3])
        return J

    def build_fuzzy_engine(self, json_path):
        with open(json_path, "r") as file:
            data = json.load(file)

        engine = fl.Engine(name=data["Name"])
        mf_type_mapping = {
            "trimf": lambda name, params: fl.Triangle(name, *params),
            "trapmf": lambda name, params: fl.Trapezoid(name, *params),
            "gaussmf": lambda name, params: fl.Gaussian(name, params[0], params[1]),
            "gbellmf": lambda name, params: fl.Bell(name, params[0], params[1], params[2]),
            "sigmf": lambda name, params: fl.Sigmoid(name, params[0], params[1]),
        }

        for input_var in data["Inputs"]:
            iv = fl.InputVariable(
                name=input_var["Name"],
                minimum=input_var["Range"][0],
                maximum=input_var["Range"][1],
                lock_range=False,
                terms=[mf_type_mapping[mf["Type"].lower()](mf["Name"], mf["Parameters"]) for mf in input_var["MembershipFunctions"]],
            )
            engine.input_variables.append(iv)

        for output_var in data["Outputs"]:
            ov = fl.OutputVariable(
                name=output_var["Name"],
                minimum=output_var["Range"][0],
                maximum=output_var["Range"][1],
                lock_range=False,
                lock_previous=False,
                default_value=0.0,
                aggregation=fl.Maximum(),
                defuzzifier=fl.Centroid(resolution=100),
                terms=[mf_type_mapping[mf["Type"].lower()](mf["Name"], mf["Parameters"]) for mf in output_var["MembershipFunctions"]],
            )
            engine.output_variables.append(ov)

        rb = fl.RuleBlock(
            name="rules",
            conjunction=None,
            disjunction=None,
            implication=fl.Minimum(),
            activation=fl.General(),
            rules=[
                fl.Rule.create(
                    "if "
                    + " and ".join(
                        f"{data['Inputs'][ante[0]]['Name']} is {data['Inputs'][ante[0]]['MembershipFunctions'][ante[1]-1]['Name']}"
                        for ante in enumerate(rule["Antecedent"])
                        if 0 <= ante[0] < len(data["Inputs"]) and 0 <= ante[1] - 1 < len(data["Inputs"][ante[0]]["MembershipFunctions"])
                    )
                    + " then "
                    + " and ".join(
                        f"{data['Outputs'][cons[0]]['Name']} is {data['Outputs'][cons[0]]['MembershipFunctions'][cons[1]-1]['Name']}"
                        for cons in enumerate(rule["Consequent"])
                        if 0 <= cons[0] < len(data["Outputs"]) and 0 <= cons[1] - 1 < len(data["Outputs"][cons[0]]["MembershipFunctions"])
                    ),
                    engine,
                )
                for rule in data["Rules"]
            ],
        )

        engine.rule_blocks.append(rb)

        return engine
        
    def imu_callback(self, msg):
        self.theta = msg.angle
        self.w1 = msg.w1
        self.w2 = msg.w2
        self.w3 = msg.w3
        self.w4 = msg.w4

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
            
    def get_inv_Jacobian(self,th):
        th1 = th + np.pi/4
        r2 = np.sqrt(2)
        J_inv = np.array([[r2 * np.cos(th1) , r2 * np.sin(th1), -(self.lx + self.ly)],
                          [r2 * np.sin(th1) ,-r2 * np.cos(th1), -(self.lx + self.ly)],
                          [r2 * np.cos(th1) , r2 * np.sin(th1),  (self.lx + self.ly)],
                          [r2 * np.sin(th1) ,-r2 * np.cos(th1),  (self.lx + self.ly)]])
        return J_inv

    def compute_control(self):
        e_x = self.target_x - self.x
        e_y = self.target_y - self.y

        self.engine.input_variable("input1").value = e_x
        self.engine.input_variable("input2").value = e_y
        self.engine.process()

        # Log active rules
        active_rules = []
        for rule_block in self.engine.rule_blocks:
            for rule in rule_block.rules:
                if rule.weight > 0:  # Log only active rules
                    active_rules.append(rule.text)

        vx = self.engine.output_variable("output1").value
        vy = self.engine.output_variable("output2").value

        return vx, vy
    
    def run(self):
        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            start_time = time.time()  # 'tic'
            vx, vy = self.compute_control()

            # Convert velocities to wheel speeds using inverse kinematics
            ac = np.array([[vx], [vy], [0]])
            J_inv = np.linalg.pinv(self.calculate_jacobian())
            U = np.dot(J_inv, ac)

            # Saturate wheel speeds
            w1, w2, w3, w4 = self.saturate_wheel_speeds(U.flatten())

            self.control_publisher.publish(Float32MultiArray(data=[w1, w2, w3, w4]))


            rate.sleep()  # Espera hasta el siguiente ciclo para cumplir con la tasa establecida

            end_time = time.time()  # 'toc'

            elapsed_time = end_time - start_time
            print("Time taken for this loop:", elapsed_time, "seconds")
            

if __name__ == "__main__":
    try:
        node = PoseControl()
        node.run()
        rospy.spin()
    except Exception as e:
        rospy.logerr(str(e))
    finally:
        #node.control_publisher.publish(Float32MultiArray(data=[0, 0, 0, 0]))
        sys.exit()
