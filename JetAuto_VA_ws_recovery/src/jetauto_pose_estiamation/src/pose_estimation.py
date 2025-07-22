#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import rospy
import tf.transformations
import math
from std_msgs.msg import Float32MultiArray
#from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
from jetauto_interfaces.msg import imu_encoder
#rom sensor_msgs.msg import


odom = Odometry()

class Compute_Vel_Pose:
    def __init__(self):
        rospy.init_node('Estimate_Pose', anonymous=False)
        #rospy.Subscriber('/filtered_angular_speed', Float32MultiArray, self.test_callback) #todos los subscriber tengan  rospy.Subscriber('/filtered_angular_speed', Float32MultiArray, self.test_callback, 100)
        rospy.Subscriber('/imu_encoder', imu_encoder, self.imu_encoder_callback)
        self.odom = Odometry()
        self.imu_encoder_msg = imu_encoder() #este mensaje se llama imu_encoder
        #self.file = open("")
        #constantes del robot
        self.r = rospy.get_param("r", 0.0485)
        self.lx = rospy.get_param("lx", 0.0975)
        self.ly = rospy.get_param("ly", 0.103)
        self.odom_pub = rospy.Publisher("/jetauto_odom", Odometry, queue_size=2)
        # para el calculo de la posicion
        self.time_ant = rospy.Time.now()
        self.vx_ant = 0
        self.vy_ant = 0
        self.wz_ant = 0
        self.px_ant = 0
        self.py_ant = 0
        self.theta_ant = 0
        self.y_imu = 0
        #self.file = open("/home/jetauto/JetAuto_VA_ws/src/Datos_Graficas/Th3.txt","w")
        #self.file.write("t\tpx\tpy\ttheta\n")
        #self.file.close()
        self.t1 = rospy.get_rostime()

        

    def imu_encoder_callback(self, msg):
        #self.file = open("dndresin.txt","w")
        #self.file.write("Hello There \n")
        #self.file.write("Hello There \n")
        #self.file.write("Hello There \n")
        
        #print("Creado")
        time_now = msg.imu.header.stamp
        dt = (time_now - self.time_ant).to_sec()
        #print(dt)
        self.time_ant = time_now
        w1 = msg.w1
        w2 = msg.w2
        w3 = msg.w3
        w4 = msg.w4
        yaw = msg.angle       # angulo en rads
        #rate = rospy.Rate(5)
	# Cinematica Inversa
        a = 4*(self.lx+self.ly)
        vx = (w1+w2+w3+w4)*self.r/4
        vy = (w1-w2+w3-w4)*self.r/4
        #wz = (-w1+w2-w3+w4)*self.r/a
        #print("\nVx: " + str(vx))
        #print("Vy: " + str(vy))
        #print("wz: " + str(wz))    
        #yaw = self.y_imu
        
        vxw = vx*math.cos(yaw)-vy*math.sin(yaw)
        vyw = vx*math.sin(yaw)+vy*math.cos(yaw)
        px = self.px_ant+(vxw+self.vx_ant)*dt/2.0
        py = self.py_ant+(vyw+self.vy_ant)*dt/2.0
        theta = yaw

        #print("\n\npx: " + str(px))
        #print("py: " + str(py))
        #print("theta: " + str(theta))
        #print("dt: " + str(dt))
        
        odom.pose.pose.position.x = px            		# agregar el header al Odometry
        odom.pose.pose.position.y = py
        quat = tf.transformations.quaternion_from_euler(0, 0, theta)
        odom.pose.pose.orientation = Quaternion(*quat)        
        self.odom_pub.publish(odom)

        #hola = open("/home/jetauto/JetAuto_VA_ws/src/Datos_Graficas/Th3.txt","a")
        #self.file.write(w1+"\t")
        #hola.write(str((time_now-self.t1).to_sec())+"\t")
        #hola.write(str(px)+"\t"+str(py)+"\t"+str(theta)+"\n")
        #hola.close()
                                                                   # Caso velocidad sin filtro
        self.vx_ant = vxw
        self.vy_ant = vyw
        #self.wz_ant = wz
        self.px_ant = px
        self.py_ant = py
        self.theta_ant = theta
        #self.file.close()


        
if __name__ == "__main__":
    try:
        node = Compute_Vel_Pose()
        
        
	
        rospy.spin()
    except Exception as e:
        rospy.logerr(str(e))
    finally:
        #node.motor_controller.set_speed((0, 0, 0, 0))
        
        sys.exit()
