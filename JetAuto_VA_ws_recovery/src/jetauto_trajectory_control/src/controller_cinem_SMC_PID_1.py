#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import rospy
import rospkg
import matplotlib.pyplot as plt
import Paths as paths
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Pose, Point, Twist
from math import sqrt
import time
from std_msgs.msg import Float32MultiArray, Header
from geometry_msgs.msg import PoseStamped
from jetauto_interfaces.msg import imu_encoder

import math
import numpy as np
from math import atan2


class PoseControl:
    def __init__(self):
        rospy.init_node("trajectory_controller", anonymous=False)
        
        path_type = rospy.get_param('path_type', 'ellipse')
        self.tm = rospy.get_param('tiempo_muestreo', 0.1)
        self.tf = rospy.get_param('tiempo_total', 80)
        self.r = rospy.get_param('r', 0.0485)
        self.lx = rospy.get_param('lx', 0.0975)
        self.ly = rospy.get_param('ly', 0.103)
        self.guardar_datos = rospy.get_param('guardar_datos', True)
        
        # Parameter for SMC
        self.KD = rospy.get_param('SMC/K_d', 0.2)
        self.delta = rospy.get_param('SMC/delta', 0.7)
        self.lambda_p = rospy.get_param('SMC/lambda_p', 1)
        self.lambda_d = rospy.get_param('SMC/lambda_d', 1)
        self.lambda_i = rospy.get_param('SMC/lambda_i', 1)
        
        rospy.Subscriber('/imu_encoder', imu_encoder, self.imu_callback)
        rospy.Subscriber('/jetauto_odom', Odometry, self.odom_callback)
        
        self.control_publisher = rospy.Publisher("wheel_setpoint", Float32MultiArray, queue_size=10)
      
        self.control_publisher.publish(Float32MultiArray(data=[0, 0, 0, 0]))
        
        self.x = 0
        self.y = 0
        self.theta = 0.0
        self.w1 = 0.0
        self.w2 = 0.0
        self.w3 = 0.0
        self.w4 = 0.0
        
        self.x_ant = 0.0
        self.y_ant = 0.0
        self.theta_ant = 0.0 
        
        self.e_x_ant = 0.0
        self.e_y_ant = 0.0
        self.e_theta_ant = 0.0
        
        self.ei_x_ant = 0.0
        self.ei_y_ant = 0.0
        self.ei_theta_ant = 0.0
        
        self.e_x_ant_2 = 0.0
        self.e_y_ant_2 = 0.0
        self.e_theta_ant_2 = 0.0

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

        #Para guardar datos en txt
        if self.guardar_datos:
            # Use rospkg to find the path to the package
            rospack = rospkg.RosPack()
            package_path = rospack.get_path('jetauto_trajectory_control')
            # Construct the directory path
            directory = os.path.join(package_path,'datos','SMC_Lyap')
            if not os.path.exists(directory):
                os.makedirs(directory)  # Create the directory if it doesnt exist
            self.file_name = os.path.join(directory, "SMC_Lyap_{}.txt".format(path_type))
            with open(self.file_name, "w") as file:
                pass
                     
        ##Definir trayectoria
        pth = paths.Paths()
        self.goalx    = pth.x
        self.goalx_d = pth.vx
        self.goaly    = pth.y
        self.goaly_d = pth.vy
        self.time     = pth.t
        self.tm       = pth.Ts
        self.goaltheta = pth.theta
        self.goaltheta_d = pth.w
        

        
        #Graficar trayectoria a seguir
        plt.figure(figsize=(10, 10))
        plt.plot(self.goalx, self.goaly)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Trayectoria a seguir')
        plt.grid(True)
        circle = plt.Circle((self.goalx[0], self.goaly[0]), radius=0.05, color='b', alpha=1)
        plt.gca().add_patch(circle)
        plt.axis('equal')  # Ensure aspect ratio is equal 
        plt.show()
        
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
        #Error x
        #plt.plot(self.t,self.x_error)
        #plt.xlabel('time')
        #plt.ylabel('x error')
        #plt.title('X Error')
        #plt.grid(True)
        #plt.show()
        
        #Error y
        #plt.plot(self.t,self.y_error)
        #plt.xlabel('time')
        #plt.ylabel('y error')
        #plt.title('Y Error')
        #plt.grid(True)
        #plt.show()
        
        #Comparacion trayectorias
        plt.figure(figsize=(win_size_y, win_size_y))
        plt.plot(self.goalx, self.goaly,label='Referencia')
        plt.plot(self.x_sim,self.y_sim,label='Simulacion')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.title('Trayectoria')
        plt.grid(True)
        plt.legend()
        circle = plt.Circle((self.x_sim[-1], self.y_sim[-1]), radius=0.05, color='tab:orange', alpha=1)
        plt.gca().add_patch(circle)
        plt.axis('equal')  # Ensure aspect ratio is equal 
        plt.show()

        plt.figure(figsize=(win_size_y, win_size_y))
        plt.plot(self.time, self.goaltheta,label='Referencia')
        plt.plot(self.time,self.theta_sim,label='Simulacion')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.title('Theta')
        plt.grid(True)
        plt.legend()
        #plt.axis('equal')  # Ensure aspect ratio is equal 
        plt.show()
        
        #Senales de Contol
        #plt.plot(self.t, self.w1,label='w1')
        #plt.plot(self.t,self.w2,label='w2')
        #plt.plot(self.t, self.w3,label='w3')
        #plt.plot(self.t,self.w4,label='w4')
        #plt.xlabel('x')
        #plt.ylabel('y')
        #plt.title('Velocidad ruedas')
        #plt.grid(True)
        #plt.legend() 
        #plt.show()
    
    def get_inv_Jacobian(self,th):
        th1 = th + np.pi/4
        r2 = np.sqrt(2)
        J_inv = np.array([[r2 * np.cos(th1) , r2 * np.sin(th1), -(self.lx + self.ly)],
                          [r2 * np.sin(th1) ,-r2 * np.cos(th1), -(self.lx + self.ly)],
                          [r2 * np.cos(th1) , r2 * np.sin(th1),  (self.lx + self.ly)],
                          [r2 * np.sin(th1) ,-r2 * np.cos(th1),  (self.lx + self.ly)]])
        return J_inv
    
    def run(self):
    
        init_time = rospy.Time.now()
        last_time = init_time
        
        for i in range(0,len(self.goalx)):
                 
            while not rospy.is_shutdown() and (rospy.Time.now()-init_time).to_sec() < self.time[i]:
                pass
            #print(1)    
            dt = (rospy.Time.now()-last_time).to_sec()
            last_time = rospy.Time.now()
            
            # Velocidad lineal y angular del robot
            vx = (self.x_ant - self.x)/self.tm
            vy = (self.y_ant - self.y)/self.tm
            w = (self.theta_ant - self.theta)/self.tm
            
            # Errores
            e_x = self.goalx[i] - self.x
            e_y = self.goaly[i] - self.y
            e_theta = self.goaltheta[i] - self.theta
            e_theta = (e_theta + np.pi) % (2*np.pi) - np.pi
            
            # Integral de los errores
            ei_x = self.ei_x_ant + e_x * dt
            ei_y = self.ei_y_ant + e_y * dt
            ei_theta = self.ei_theta_ant + e_theta * dt
            
            # Derivada de los errores
            ed_x = (e_x - self.e_x_ant)/self.tm
            ed_y = (e_y - self.e_y_ant)/self.tm
            ed_theta = (e_theta - self.e_theta_ant)/self.tm
                                  
            # Vectores
            e_v = np.array([[e_x],[e_y],[e_theta]])
            ei_v = np.array([[ei_x],[ei_y],[ei_theta]])
            ed_v = np.array([[ed_x],[ed_y],[ed_theta]])
            ref_d = np.array([[self.goalx_d[i]],[self.goaly_d[i]],[self.goaltheta_d[i]]])
            
            #Superficie deslizante PD
            s = self.lambda_p*e_v +self.lambda_i*ei_v
            s_1 = s / (np.abs(s) + self.delta)
            
            aC = ref_d + (self.lambda_i/self.lambda_p)*e_v
            aD = (self.KD/self.lambda_p)*s_1
            
            #Calculo del jacobiano
            J_inv = self.get_inv_Jacobian(self.theta)
            w = np.dot(J_inv,(aC + aD))/self.r
            
            w1_aux = w[0,0]
            w2_aux = w[1,0]
            w3_aux = w[2,0]
            w4_aux = w[3,0]
            a = 9.00
            w1 = max(min(w1_aux, a), -a)
            w2 = max(min(w2_aux, a), -a)
            w3 = max(min(w3_aux, a), -a)
            w4 = max(min(w4_aux, a), -a)
            
           
            
            # Datos i-1
            self.e_x_ant = e_x
            self.e_y_ant = e_y
            self.e_theta_ant = e_theta
            
            self.ei_x_ant = ei_x
            self.ei_y_ant = ei_y
            self.ei_theta_ant = ei_theta
            
            self.x_ant = self.x
            self.y_ant = self.y
            self.theta_ant = self.theta
            
            self.append_data((last_time-init_time).to_sec(),self.x,self.y,self.theta,self.w1,self.w2,self.w3,self.w4,w1,w2,w3,w4)
            
            # Publish the wheels message
            self.control_publisher.publish(Float32MultiArray(data=[w1, w2, w3, w4]))
            
            #Append pose sim para graficar en RVIZ
            self.pose = PoseStamped()
            self.pose.header = Header()
            self.pose.header.stamp = rospy.Time.now()
            self.pose.header.frame_id = 'odom'
            self.pose.pose.position.x = self.x
            self.pose.pose.position.y = self.y
            self.pose.pose.position.z = 0
            self.pose.pose.orientation.w = 1.0  # No rotation, quaternion format
            #self.path_sim.poses.append(self.pose)
                        
        self.control_publisher.publish(Float32MultiArray(data=[0, 0, 0, 0]))
        if self.guardar_datos:
                self.write_file(self.t,self.goalx, self.goaly, self.goaltheta, self.x_sim, self.y_sim, self.theta_sim, self.w1_sim, self.w2_sim, self.w3_sim, self.w4_sim, self.w1_ref, self.w2_ref, self.w3_ref, self.w4_ref)
            
        self.plot()

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
