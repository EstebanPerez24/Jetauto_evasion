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
        self.tm = rospy.get_param('tiempo_muestreo', '0.05')
        self.tf = rospy.get_param('tiempo_total', '60')
        self.r = rospy.get_param('r', '0.0485')
        self.lx = rospy.get_param('lx', '0.0975')
        self.ly = rospy.get_param('ly', '0.103')
        self.guardar_datos = rospy.get_param('guardar_datos', True)
        path_type = ""
        
        self.control_publisher = rospy.Publisher("wheel_setpoint", Float32MultiArray, queue_size=10)
        
        rospy.Subscriber('/imu_encoder', imu_encoder, self.imu_callback)
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
            self.file_name = os.path.join(directory, "COMP_{}.txt".format(path_type))
            with open(self.file_name, "w") as file:
                pass
            
        #Iniciar posicion del robot
        
        self.control_publisher.publish(Float32MultiArray(data=[0, 0, 0, 0]))
        
        rospy.sleep(1.5)
      
        self.x = 0
        self.y = 0
        self.theta = 0.0
        self.w1 = 0.0
        self.w2 = 0.0
        self.w3 = 0.0
        self.w4 = 0.0
        
        self.t = []
        self.x_sim = []
        self.y_sim = []
        self.theta_sim = []
        self.w1_sim = []
        self.w2_sim = []
        self.w3_sim = []
        self.w4_sim = []
        self.w1_ref = []
        self.w2_ref = []
        self.w3_ref = []
        self.w4_ref = []             
                     
        ##Definir trayectoria
        pth = paths.Paths()
        self.goalx    = pth.x
        self.goalx_d = pth.vx
        self.goaly    = pth.y
        self.goaly_d = pth.vy
        self.time     = pth.t
        self.tm      = pth.Ts
        self.goaltheta = pth.theta
        self.goaltheta_d = pth.w
        

    def imu_callback(self, msg):
        self.theta = msg.angle
        self.w1 = msg.w1
        self.w2 = msg.w2
        self.w3 = msg.w3
        self.w4 = msg.w4

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        
    def append_data(self, t,x,y,theta,w1_sim,w2_sim,w3_sim,w4_sim,w1_ref,w2_ref,w3_ref,w4_ref):
        self.t.append(t)
        self.x_sim.append(x)
        self.y_sim.append(y)
        self.theta_sim.append(theta)
        self.w1_sim.append(w1_sim)
        self.w2_sim.append(w2_sim)
        self.w3_sim.append(w3_sim)
        self.w4_sim.append(w4_sim)
        self.w1_ref.append(w1_ref)
        self.w2_ref.append(w2_ref)
        self.w3_ref.append(w3_ref)
        self.w4_ref.append(w4_ref)

    def write_file(self, t, x, y, theta, x_sim, y_sim, theta_sim, w1_sim, w2_sim, w3_sim, w4_sim, w1_ref, w2_ref, w3_ref, w4_ref):
        with open(self.file_name, "a") as file:
            for i in range(0,len(self.goalx)): 
                file.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(t[i], x[i], y[i], theta[i], x_sim[i], y_sim[i], theta_sim[i], w1_sim[i], w2_sim[i], w3_sim[i], w4_sim[i], w1_ref[i], w2_ref[i], w3_ref[i], w4_ref[i]))
        
    def plot(self):
        win_size_x = 15
        win_size_y = 10             
               
        #Comparacion trayectorias
        plt.figure(figsize=(win_size_y, win_size_y))
        plt.plot(self.x_sim,self.y_sim)
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.title('Trayectoria')
        plt.grid(True)
        plt.legend()
        plt.axis('equal')  # Ensure aspect ratio is equal 
        plt.show() 
        
        #Senales de control
        plt.figure(figsize=(win_size_x, win_size_y))
        plt.plot(self.t, self.w1_sim,label='w1',linewidth=0.5)
        plt.plot(self.t, self.w2_sim,label='w2',linewidth=0.5)
        plt.plot(self.t, self.w3_sim,label='w3',linewidth=0.5)
        plt.plot(self.t, self.w4_sim,label='w4',linewidth=0.5)
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Velocidad [rad/s]')
        plt.title('Senal de control ruedas')
        plt.grid(True)
        plt.legend()
        plt.show() 
    
    def run(self):
        ref_theta = 0
        init_time = rospy.Time.now()
        last_time = init_time
        a = 2.5
        stop_length = 0.05  #porcentaje
        setpoint = Float32MultiArray(data=[0, 0, 0, 0])
        for i in range(0,len(self.time)):
            while not rospy.is_shutdown() and (rospy.Time.now()-init_time).to_sec() < self.time[i]:
                pass

            last_time = rospy.Time.now()
            if i > (len(self.time)*3.0/4.0):
                if i > ((1-stop_length)*len(self.time)):
                    setpoint = Float32MultiArray(data=[0, 0, 0, 0])
                else:
                    setpoint = Float32MultiArray(data=[a, -a, a, -a])
            elif i > (len(self.time)*2.0/4.0):
                if i > ((3.0/4.0-stop_length)*len(self.time)):
                    setpoint = Float32MultiArray(data=[0, 0, 0, 0])
                else:
                    setpoint = Float32MultiArray(data=[-a, -a, -a, -a])
            elif i > (len(self.time)*1.0/4.0):
                if i > ((2.0/4.0-stop_length)*len(self.time)):
                    setpoint = Float32MultiArray(data=[0, 0, 0, 0])
                else:
                    setpoint = Float32MultiArray(data=[-a, a, -a, a])
            else:
                if i > ((1.0/4.0-stop_length)*len(self.time)):
                    #print("Cero")
                    setpoint = Float32MultiArray(data=[0, 0, 0, 0])
                else:
                    setpoint = Float32MultiArray(data=[a, a, a, a])
            setpoint = Float32MultiArray(data=[1, 2, 3, 4])
            #print(i,(1/4-stop_length),len(self.time))
            #print(setpoint,i)
            # Publish the stop message
            self.control_publisher.publish(setpoint)
                          
            self.append_data((last_time-init_time).to_sec(),self.x,self.y,self.theta,self.w1,self.w2,self.w3,self.w4,setpoint.data[0],setpoint.data[1],setpoint.data[2],setpoint.data[3])
            
        if self.guardar_datos:
            self.write_file(self.t,self.goalx, self.goaly, self.goaltheta, self.x_sim, self.y_sim, self.theta_sim, self.w1_sim, self.w2_sim, self.w3_sim, self.w4_sim, self.w1_ref, self.w2_ref, self.w3_ref, self.w4_ref)

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
