#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import rospy
import rospkg
import matplotlib.pyplot as plt
import Paths as paths
from nav_msgs.msg import Odometry
#from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Pose, Point, Twist
from math import sqrt
import time
from std_msgs.msg import Float32MultiArray, Header
from geometry_msgs.msg import PoseStamped
from jetauto_interfaces.msg import imu_encoder
import fuzzylite as fl
import json
import math
import numpy as np
from math import atan2


class PoseControl:
    def __init__(self):
        rospy.init_node("trajectory_controller", anonymous=False)

        # Load fuzzy engines from JSON
        self.engine_vx = self.build_fuzzy_engine("/home/jetauto/JetAuto_VA_ws/src/jetauto_trajectory_control/src/anfis_vx.json")
        self.engine_vy = self.build_fuzzy_engine("/home/jetauto/JetAuto_VA_ws/src/jetauto_trajectory_control/src/anfis_vy.json")
        self.engine_w  = self.build_fuzzy_engine("/home/jetauto/JetAuto_VA_ws/src/jetauto_trajectory_control/src/anfis_omega.json")
        
        path_type = rospy.get_param('path_type', 'ellipse')
        self.tm = rospy.get_param('tiempo_muestreo', 0.1)
        self.tf = rospy.get_param('tiempo_total', 80)
        self.r = rospy.get_param('r', 0.0485)
        self.lx = rospy.get_param('lx', 0.0975)
        self.ly = rospy.get_param('ly', 0.103)
        self.guardar_datos = rospy.get_param('guardar_datos', True)

        self.usar_fuzzy = False
        self.aux = False
        self.saved_index = 0
        self.fixed_index = 0
        self.indice_encontrado = 0
        self.closest_centroids = (1.0, 360.0)
        self.write_counter = 0
        
        # Parameter for SMC
        self.tau_v = rospy.get_param('SMC_TF/tau_v', 0.98)
        self.tau_w = rospy.get_param('SMC_TF/tau_w', 1.4)
        self.K_v = rospy.get_param('SMC_TF/K_v', 1)
        self.K_w = rospy.get_param('SMC_TF/K_w', 1.21)
        self.KD = rospy.get_param('SMC_TF/K_d', 0.2)
        self.lambda_0 = rospy.get_param('SMC_TF/lambda_0', 1)
        self.delta = rospy.get_param('SMC_TF/delta', 0.7)
        self.lambda_1v = 1/self.tau_v
        self.lambda_1w = 1/self.tau_w
        
        rospy.Subscriber('/imu_encoder', imu_encoder, self.imu_callback)
        rospy.Subscriber('/jetauto_odom', Odometry, self.odom_callback)
        rospy.Subscriber('/closest_centroids', Float32MultiArray, self.closest_centroid_callback)        
        
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
            directory = os.path.join(package_path,'datos','SMC_TF_ANFIS')
            if not os.path.exists(directory):
                os.makedirs(directory)  # Create the directory if it doesnt exist
            self.file_name = os.path.join(directory, "SMC_TF_ANFIS_{}.txt".format(path_type))
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

    def closest_centroid_callback(self, msg):
        try:
            distance = msg.data[0]  # Distance to the closest obstacle
            angle = msg.data[1]     # Angle to the closest obstacle
            self.closest_centroids = (distance, angle)
        except Exception as e:
            rospy.logerr("Error in closest_centroid_callback:",e)

        
    def build_fuzzy_engine(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)

        print("Creando motor difuso:", data['engine'])
        engine = fl.Engine()
        engine.name = data['engine']

        # Entradas
        print("Procesando variables de entrada...")
        for input_var in data["inputs"]:
            var = fl.InputVariable()
            var.name = input_var["name"]
            var.enabled = True
            var.range = input_var["range"]
            for term in input_var["terms"]:
                if term["type"].lower() in ["gaussmf", "gaussian"]:
                    t = fl.Gaussian(term["name"], term["parameters"][1], term["parameters"][0])
                else:
                    raise ValueError(f"Tipo de MF no soportado: {term['type']}")
                var.terms.append(t)
            engine.input_variables.append(var)

        # Salida
        print("Procesando variable de salida...")
        output = data["outputs"]
        var_out = fl.OutputVariable()
        var_out.name = output["name"]
        var_out.range = output["range"]
        var_out.default_value = 0.0
        var_out.defuzzifier = fl.WeightedAverage()
        var_out.lock_valid_output = False
        var_out.lock_output_range = False

        for term in output["terms"]:
            if term["type"].lower() == "linear":
                t = fl.Linear(term["name"], term["parameters"])
                t.engine = engine  # ← Agrega referencia al motor difuso
            else:
                raise ValueError(f"Tipo de MF de salida no soportado: {term['type']}")
            var_out.terms.append(t)


        engine.output_variables.append(var_out)

        # Reglas
        print("Procesando reglas...")
        rule_block = fl.RuleBlock()
        rule_block.name = "rules"
        rule_block.enabled = True
        rule_block.conjunction = fl.AlgebraicProduct()
        rule_block.disjunction = fl.Maximum()
        rule_block.implication = fl.AlgebraicProduct()
        rule_block.activation = fl.General()  # ← necesario

        for rule in data["rules"]:
            ant = " and ".join([f"in{i+1} is in{i+1}cluster{a}" for i, a in enumerate(rule["antecedent"])])
            cons = f"{output['name']} is out1cluster{rule['consequent']}"
            rule_str = f"if {ant} then {cons} with {rule['weight']}"
            rule_block.rules.append(fl.Rule.create(rule_str, engine))

        engine.rule_blocks.append(rule_block)

        return engine

    def mover_y_encontrar_indice(self):
        # 1. Encontrar el índice del punto más cercano de la trayectoria
        distancias = np.sqrt((np.array(self.goalx) - self.x)**2 + (np.array(self.goaly) - self.y)**2)
        indice_cercano = np.argmin(distancias)
        
        x1 = self.goalx[indice_cercano]
        y1 = self.goaly[indice_cercano]

        # Retornamos solo el índice cercano y la posición
        return indice_cercano, x1, y1
        
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
            for i in range(0, len(self.t)):  # ← usa la longitud real de datos almacenados
                file.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                    t[i], x[i], y[i], theta[i],
                    x_sim[i], y_sim[i], theta_sim[i],
                    w1_sim[i], w2_sim[i], w3_sim[i], w4_sim[i],
                    w1_ref[i], w2_ref[i], w3_ref[i], w4_ref[i]
                ))

        
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

    def append_line_to_file(self):
        if not self.guardar_datos:
            return
        self.write_counter += 1
        if self.write_counter % 25 != 0:  # ← Solo cada 10 ciclos
            return
        with open(self.file_name, "a") as file:
            i = len(self.t) - 1
            if i >= 0:
                file.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                    self.t[i], self.goalx[min(i, len(self.goalx)-1)], self.goaly[min(i, len(self.goaly)-1)], self.goaltheta[min(i, len(self.goaltheta)-1)],
                    self.x_sim[i], self.y_sim[i], self.theta_sim[i],
                    self.w1_sim[i], self.w2_sim[i], self.w3_sim[i], self.w4_sim[i],
                    self.w1_ref[i], self.w2_ref[i], self.w3_ref[i], self.w4_ref[i]
                ))
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
        
        i = 0
        while i < len(self.goalx) and not rospy.is_shutdown():
                 
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
            
            #Superficie deslizante PID de velocidad lineal
            svx = self.lambda_1v*e_x + self.lambda_0*ei_x + ed_x
            s_vx = svx / (np.abs(svx) + self.delta)
            
            svy = self.lambda_1v*e_y + self.lambda_0*ei_y + ed_y
            s_vy = svy / (np.abs(svy) + self.delta)
            
            #Superficie deslizante PID de velocidad angular
            sw = self.lambda_1v*e_theta + self.lambda_0*ei_theta + ed_theta
            s_w = sw / (np.abs(sw) + self.delta)
            
            s_1 = np.array([[s_vx],[s_vy],[s_w]])

            # Activar fuzzy si hay obstáculo cercano
            if self.closest_centroids[0] < 0.30:

                if not self.usar_fuzzy:
                    self.fixed_index = i 
                    rospy.loginfo("Activando control fuzzy")
                    self.usar_fuzzy = True
                    self.aux = False

            if self.usar_fuzzy:
            
                inputs = self.closest_centroids # [in1, in2]
                    
                for j in range(2):
                    self.engine_vx.input_variable(f"in{j+1}").value = self.closest_centroids[j]
                    self.engine_vy.input_variable(f"in{j+1}").value = self.closest_centroids[j]
                    self.engine_w.input_variable(f"in{j+1}").value = self.closest_centroids[j]
                    
                self.engine_vx.process()
                self.engine_vy.process()
                self.engine_w.process()
                
                # Salidas del fuzzy en marco del LIDAR (fijo)
                vx_lidar = self.engine_vx.output_variable("out1").value
                vy_lidar = self.engine_vy.output_variable("out1").value
                acw = -self.engine_w.output_variable("out1").value

                # Rotar al marco del robot usando theta
                acx = vx_lidar * np.cos(self.theta) - vy_lidar * np.sin(self.theta)
                acy = vx_lidar * np.sin(self.theta) + vy_lidar * np.cos(self.theta)

                ac_vector = np.array([[acx],[acy],[acw]])
                
                # Buscar índice para reanudar
                min_dist = float('inf')
                closest_index = -1

                for k in range(self.fixed_index + 20, len(self.goalx)):
                    dist = math.sqrt((self.x - self.goalx[k])**2 + (self.y - self.goaly[k])**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_index = k

                if min_dist < 0.05:
                    rospy.loginfo("Reanudando control de trayectoria en el punto más cercano")
                    self.control_publisher.publish(Float32MultiArray(data=[0, 0, 0, 0]))
                    print("Índice trayectoria más cercano:", closest_index)
                    i = closest_index
                    self.usar_fuzzy = False

                    self.e_x_ant = 0
                    self.e_y_ant = 0
                    self.e_theta_ant = 0
                    ei_x = 0
                    ei_y = 0
                    ei_theta = 0

                    init_time = rospy.Time.now() - rospy.Duration.from_sec(self.time[i])

                    # Esperar que el tiempo actual alcance el tiempo del nuevo índice
                    while not rospy.is_shutdown() and (rospy.Time.now() - init_time).to_sec() < self.time[i]:
                        pass

                if self.usar_fuzzy and (self.closest_centroids[0] == 1.0 and self.closest_centroids[1] == 360.0):
                    rospy.loginfo("Desactivando control fuzzy (obstáculo lejano o sin detección)")
                    
                    if not self.aux:
                        self.fixed_index, x1, y1 = self.mover_y_encontrar_indice()

                        if i != self.fixed_index:
                            rospy.loginfo(f"Reanudando control de trayectoria en el nuevo índice: {self.fixed_index}")
                            i = self.fixed_index
                            init_time = rospy.Time.now() - rospy.Duration.from_sec(self.time[i])
                        else:
                            rospy.loginfo("Ya estás en el índice correcto, no se necesita reanudar")
                        
                        self.aux = True  # ya reanudó

                    self.usar_fuzzy = False
            else:   
            
                # Acciones de control continua
                vx_ac = self.tau_v*(vx*(1/(self.tau_v) - self.lambda_1v) + self.lambda_1v * self.goalx_d[i] + self.lambda_0 * e_x)
                vy_ac = self.tau_v*(vy*(1/(self.tau_v) - self.lambda_1v) + self.lambda_1v * self.goaly_d[i] + self.lambda_0 * e_y)
                w_ac =  self.tau_w*(w*(1/(self.tau_w) - self.lambda_1w) + self.lambda_1w * self.goaltheta_d[i] + self.lambda_0* 5 * e_theta)
                
                ac_vector = np.array([[vx_ac],[vy_ac],[w_ac]]) + self.KD * s_1
                i += 1 
                pass

            #Calculo del jacobiano
            J_inv = self.get_inv_Jacobian(self.theta)
            w = np.dot(J_inv,ac_vector)/self.r
            
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

            #print(w1)
            self.append_data((last_time-init_time).to_sec(),self.x,self.y,self.theta,self.w1,self.w2,self.w3,self.w4,w1,w2,w3,w4)
            self.append_line_to_file()
            
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
            
        if i > 599:    
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
