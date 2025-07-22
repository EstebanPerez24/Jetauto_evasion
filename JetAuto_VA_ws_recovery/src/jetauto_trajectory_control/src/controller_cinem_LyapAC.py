#!/usr/bin/env python

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

class PoseControl:
    def __init__(self):
        rospy.init_node("trajectory_controller", anonymous=False)

        #Load parameters
        self.kp = rospy.get_param('cinem_Lyap/kp', 1.9)
        path_type = rospy.get_param('path_type', 'rectangle')
        self.tm = rospy.get_param('tiempo_muestreo', '0.1')
        self.tf = rospy.get_param('tiempo_total', '60')
        self.r = rospy.get_param('r', '0.0485')
        self.lx = rospy.get_param('lx', '0.0975')
        self.ly = rospy.get_param('ly', '0.103')
        self.guardar_datos = rospy.get_param('guardar_datos', False)
                          
        rospy.Subscriber('/jetauto_odom', Odometry, self.odom_callback)

        self.control_publisher = rospy.Publisher("wheel_setpoint", Float32MultiArray, queue_size=10)
              
        #Para guardar datos en txt
        if self.guardar_datos:
            # Use rospkg to find the path to the package
            rospack = rospkg.RosPack()
            package_path = rospack.get_path('jetauto_trajectory_control')
            # Construct the directory path
            directory = os.path.join(package_path,'datos','Lyap')
            if not os.path.exists(directory):
                os.makedirs(directory)  # Create the directory if it doesnt exist
            self.file_name = os.path.join(directory, "Lyap_{}.txt".format(path_type))
            with open(self.file_name, "w") as file:
                pass
        
        
        #Iniciar posicion del robot
        
        self.control_publisher.publish(Float32MultiArray(data=[0, 0, 0, 0]))
        
        rospy.sleep(1.5)
        
            
        self.x = 0
        self.y = 0
        self.theta = 0.0
        self.e_x_ant = 0
        self.e_y_ant = 0
        self.e_theta_ant = 0.0
        self.x_error = []
        self.y_error = []
        self.theta_error = []
        self.x_sim = []
        self.y_sim = []
        self.w1 = []
        self.w2 = []
        self.w3 = []
        self.w4 = []
        self.t = []
                     
        ##Definir trayectoria
        pth = paths.Paths(self.tm, self.tf)
        self.time = pth.time()
        
        if path_type == 'ellipse':
            #Circulo/Elipse
            a = 1   #eje principal (x)
            b = 1   #ejesecundario (y)
            center = [0, -b]   # centro
            self.goalx, self.goaly = pth.ellipse(a, b, center)
            self.goalx_d, self.goaly_d = pth.ellipse_d(a, b, center)
        elif path_type == 'lemniscate':
            #Lemniscate/Forma 8
            a = 1   #factor de escalamiento
            center = [0, 0]
            self.goalx, self.goaly = pth.lemniscate(a, center)
            self.goalx_d, self.goaly_d = pth.lemniscate_d(a, center)
        elif path_type == 'spiral':
            #Espiral
            a = 0.4   #radio inicial
            b = 0.04   #razon de crecimiento
            center = [0, -a]
            self.goalx, self.goaly = pth.spiral(a, b, center)
            self.goalx_d, self.goaly_d = pth.spiral_d(a, b, center)
        elif path_type == 'line':
            #Linea
            start = [0, 0]   #coordenada inicial
            end = [3, 5]     #coordenada final
            self.goalx, self.goaly = pth.line(start, end)
            self.goalx_d, self.goaly_d = pth.line_d(start, end)
        elif path_type == 'sine':
            #Sine
            A = 1     #amplitud
            f = 1/self.tf   #frecuencia
            n = 5     # atenuacion de crecimiento en x (center_x + t/n)
            center = [0, 0]    #centro
            self.goalx, self.goaly = pth.sine(A, f*self.tf, n ,center)
            self.goalx_d, self.goaly_d = pth.sine_d(A, f*self.tf, n ,center)
        elif path_type == 'rectangle':
            #Rectangulo
            length = 3   #base (x)
            width = 2    #altura (y)
            center = [0, 0]   # centro
            self.goalx, self.goaly = pth.rectangle(length,width,center)
            self.goalx_d, self.goaly_d= pth.rectangle_d(length,width,center)
        else:
            rospy.logwarn("Tipo de trayectoria desconocida: %s" % path_type)
            self.goalx, self.goaly = [], []
        
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

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        
        rot_q = msg.pose.pose.orientation
        (roll, pitch, self.theta) = euler_from_quaternion([rot_q.x, rot_q.y, rot_q.z, rot_q.w])                        

    def calculate_ISE(self, errors):
        # Square each error value
        squared_errors = np.square(errors)
        # Integrate the squared errors (numerical integration using the trapezoidal rule)
        ise = np.trapz(squared_errors)
        return ise
        
    def write_file(self, t, x, y, theta, x_sim, y_sim, theta_sim, e_x, e_y, e_theta, w1, w2, w3, w4):
        with open(self.file_name, "a") as file:
            file.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(t, x, y, theta, x_sim, y_sim, theta_sim, e_x, e_y, e_theta, w1, w2, w3, w4))

            file.write("\n")
        
    def plot(self):
        win_size_x = 15
        win_size_y = 10      
        
        ISE_X = self.calculate_ISE(self.x_error)
        ISE_Y = self.calculate_ISE(self.y_error)
        ISE_Theta = self.calculate_ISE(self.theta_error)
        
        
               
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
        
        #Senales de control
        plt.figure(figsize=(win_size_x, win_size_y))
        plt.plot(self.t, self.w1,label='w1',linewidth=0.5)
        plt.plot(self.t, self.w2,label='w2',linewidth=0.5)
        plt.plot(self.t, self.w3,label='w3',linewidth=0.5)
        plt.plot(self.t, self.w4,label='w4',linewidth=0.5)
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
             
        for i in range(0,len(self.goalx)):
            
            if i > 0:
                #while not rospy.is_shutdown() and (rospy.Time.now()-last_time).to_sec() < self.tm:
                while not rospy.is_shutdown() and (rospy.Time.now()-init_time).to_sec() < self.time[i]:
                    pass 
 
            last_time = rospy.Time.now()
            
            
            e_x = self.goalx[i] - self.x
            e_y = self.goaly[i] - self.y
            e_theta = ref_theta - self.theta
            
            acx = (self.kp * e_x + self.goalx_d[i])
            acy = (self.kp * e_y + self.goaly_d[i])
            acw = (self.kp * e_theta)
            
            self.e_x_ant = e_x
            self.e_y_ant = e_y
            self.e_theta_ant = e_theta
            
            self.x_sim.append(self.x)
            self.y_sim.append(self.y)          
            
            theta_1 = self.theta + np.pi / 4
            J1 = [np.sqrt(2)*np.cos(theta_1),np.sqrt(2)*np.sin(theta_1),np.sqrt(2)*np.cos(theta_1),np.sqrt(2)*np.sin(theta_1)]
            J2 = [np.sqrt(2)*np.sin(theta_1),-np.sqrt(2)*np.cos(theta_1),np.sqrt(2)*np.sin(theta_1),-np.sqrt(2)*np.cos(theta_1)]
            J3 = [-1/(self.lx + self.ly), -1/(self.lx + self.ly), 1/(self.lx + self.ly), 1/(self.lx + self.ly)]
            J_array = np.array([J1,J2,J3])
            J = (self.r/4) * J_array
            J_inv = np.linalg.pinv(J)
            
            ac_vector = np.array([[acx],[acy],[acw]])
            
            w = np.dot(J_inv,ac_vector)/self.r        
            
            w1 = w[0,0]
            w2 = w[1,0]
            w3 = w[2,0]
            w4 = w[3,0]

            w1 = max(min(w1, 9), -9)
            w2 = max(min(w2, 9), -9)
            w3 = max(min(w3, 9), -9)
            w4 = max(min(w4, 9), -9)
            
            self.w1.append(w1)
            self.w2.append(w2)
            self.w3.append(w3)
            self.w4.append(w4)
                            
            # Publish the stop message
            self.control_publisher.publish(Float32MultiArray(data=[w1, w2, w3, w4]))
                          
            print([e_x,e_y,(last_time-init_time).to_sec()])
            self.x_error.append(e_x)
            self.y_error.append(e_y)
            self.theta_error.append(e_theta)
            self.t.append((last_time-init_time).to_sec())
            
            if self.guardar_datos:
                self.write_file((last_time-init_time).to_sec(),self.goalx[i], self.goaly[i], ref_theta, self.x, self.y, self.theta, e_x, e_y, e_theta, w1, w2, w3, w4)
            
            
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
