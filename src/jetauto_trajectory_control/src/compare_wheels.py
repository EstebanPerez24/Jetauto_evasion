#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import rospy
import rospkg
import matplotlib.pyplot as plt
import Paths as paths
import numpy as np
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Pose, Point, Twist
from math import sqrt
import time
from std_msgs.msg import Float32MultiArray, Header
from rosgraph_msgs.msg import Clock
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from jetauto_interfaces.msg import imu_encoder

class PoseControl:
    def __init__(self):
        rospy.init_node("trajectory_controller", anonymous=False)
        
        # Load the parameters
        self.tf = 25
        self.tm = 0.1
        self.r = rospy.get_param('r', '0.0485')
        self.lx = rospy.get_param('lx', '0.0975')
        self.ly = rospy.get_param('ly', '0.103')
        self.guardar_datos = False
        path_type = ""
        
        self.control_publisher = rospy.Publisher("wheel_setpoint", Float32MultiArray, queue_size=10)
        
        rospy.Subscriber("/imu_encoder", imu_encoder, self.control_callback)
        
        rospy.Subscriber('/jetauto_odom', Odometry, self.odom_callback)
            
            
        #Para guardar datos en txt
        if self.guardar_datos:
            # Use rospkg to find the path to the package
            rospack = rospkg.RosPack()
            package_path = rospack.get_path('jetauto_trajectory_control')
            # Construct the directory path
            directory = os.path.join(package_path,'datos','COMP')
            if not os.path.exists(directory):
                os.makedirs(directory)  # Create the directory if it doesnt exist
            self.file_name = os.path.join(directory, "COMP_ang{}.txt".format(path_type))
            with open(self.file_name, "w") as file:
                pass
            
        #Iniciar posicion del robot
        
        self.control_publisher.publish(Float32MultiArray(data=[0, 0, 0, 0]))
        
        rospy.sleep(1.5)
      
        self.theta = 0.0
        self.theta_vect = []
        self.t = []

        ##Definir trayectoria
        self.time = np.arange(0, self.tf, self.tm)
        

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        
        rot_q = msg.pose.pose.orientation
        (roll, pitch, self.theta) = euler_from_quaternion([rot_q.x, rot_q.y, rot_q.z, rot_q.w])
        
    def control_callback(self, msg):
        # Read current wheel speeds
        self.current_speeds = [msg.w1, msg.w2, msg.w3, msg.w4]
        
        

    def write_file(self, t, theta):
        with open(self.file_name, "a") as file:
            file.write("{}\t{}\n".format(t,theta))
        
    def plot(self):
        win_size_x = 15
        win_size_y = 10             
               
        #Comparacion trayectorias
        plt.figure(figsize=(win_size_y, win_size_y))
        plt.plot(self.t,self.theta_vect)
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.title('Trayectoria')
        plt.grid(True)
        plt.legend()
        plt.axis('equal')  # Ensure aspect ratio is equal 
        plt.show() 
    
    def run(self):
        self.tf = 20.0
        self.tm = 0.1
        ang = 1.0*2.0*2.0*np.pi/self.tf
        a = ang*(self.lx+self.ly)/self.r
        print(a)
        init_time = rospy.Time.now()
        last_time = init_time
        setpoint = Float32MultiArray(data=[0, 0, 0, 0])
        for i in range(0,len(self.time)):
            while not rospy.is_shutdown() and (rospy.Time.now()-init_time).to_sec() < self.time[i]:
                pass

            last_time = rospy.Time.now()
            setpoint = Float32MultiArray(data=[-a, -a, a, a])
            self.control_publisher.publish(setpoint)
                          
            #print([setpoint,i])
            self.theta_vect.append(self.theta)
            self.t.append((last_time-init_time).to_sec())
            
            if self.guardar_datos:
                self.write_file((last_time-init_time).to_sec(),self.theta)

        self.control_publisher.publish(Float32MultiArray(data=[0, 0, 0, 0]))
        self.plot()

if __name__ == "__main__":
    try:
        node = PoseControl()
        node.run()
        rospy.spin()
    except Exception as e:
        rospy.logerr(str(e))
    finally:
        node.control_publisher.publish(Float32MultiArray(data=[0, 0, 0, 0]))
        sys.exit()
