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

        # Cargar parámetros de trayectoria desde YAML
        self.trajectory_name = rospy.get_param('path_type', 'rectangle')
        parameter_path = "{}/parameters".format(self.trajectory_name)
        self.params = rospy.get_param(parameter_path, None)

        if self.params is None:
            rospy.logerr("Parámetros de", self.trajectory_name, "no encontrados en trajectory_params.yaml")
            rospy.signal_shutdown("No se pudo cargar la trayectoria")
        
        path_type = rospy.get_param('path_type', 'ellipse')
        self.tm = rospy.get_param('tiempo_muestreo', 0.1)
        self.tf = rospy.get_param('tiempo_total', 80)
        self.r = rospy.get_param('r', 0.0485)
        self.lx = rospy.get_param('lx', 0.0975)
        self.ly = rospy.get_param('ly', 0.103)
        self.guardar_datos = rospy.get_param('guardar_datos', True)
        
        # Parameter for Null Space
        self.Kp = rospy.get_param('NS/Kp', 1.5)

        # APF and control parameters
        self.zeta = 1.1547
        self.eta = 0.5
        self.dstar = 1.0 #0.5, 1.0
        self.Qstar = 0.22 #0.23, 0.22
        self.v_max = 0.125 #0.205,0.125
        self.w_max = 6.5
        self.position_accuracy = 0.05 #0.13 cuadrado, 0.05 lemniscate
        self.aux_apf = 0
        self.fixed_index = 0

        # Inicializar variables de evasión
        self.evasion_goal = None  # Punto futuro de evasión
        self.evasion_start_index = None  # Índice en el que comenzó la evasión
        self.using_apf = False  # Bandera para indicar si APF está activo
        self.kp_theta = 1.5

        # Variables de evasión
        self.start_x = 0.0  # Inicializar start_x para evitar errores
        self.start_y = 0.0  # Inicializar start_y para evitar errores
        self.max_obstacle_width = 0.5  
        self.max_obstacle_height = 0.5
        
        rospy.Subscriber('/imu_encoder', imu_encoder, self.imu_callback)
        rospy.Subscriber('/jetauto_odom', Odometry, self.odom_callback)
        rospy.Subscriber('/filtered_lines', Float32MultiArray, self.filtered_lines_callback)
        rospy.Subscriber('/obstacle_dimensions', Float32MultiArray, self.obstacle_info_callback)
        
        self.control_publisher = rospy.Publisher("wheel_setpoint", Float32MultiArray, queue_size=10)
      
        self.control_publisher.publish(Float32MultiArray(data=[0, 0, 0, 0]))
        
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
        self.hough_lines = []

        #Para guardar datos en txt
        if self.guardar_datos:
            # Use rospkg to find the path to the package
            rospack = rospkg.RosPack()
            package_path = rospack.get_path('jetauto_trajectory_control')
            # Construct the directory path
            directory = os.path.join(package_path,'datos','NS_Lyap_APF')
            if not os.path.exists(directory):
                os.makedirs(directory)  # Create the directory if it doesnt exist
            self.file_name = os.path.join(directory, "NS_Lyap_APF_{}.txt".format(path_type))
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

    def filtered_lines_callback(self, msg):
        lines = []
        data = msg.data
        if len(data) % 4 == 0:  # Asegurar que los datos lleguen en múltiplos de 4
            for i in range(0, len(data), 4):
                x1, y1, x2, y2 = data[i:i+4]
                lines.append((x1, y1, x2, y2))
                self.hough_lines = lines

    def obstacle_info_callback(self, msg):
        """
        Callback que recibe la información de las dimensiones de múltiples obstáculos,
        acumulando el tamaño máximo detectado y evitando que disminuya artificialmente.
        """
        num_obstacles = len(msg.data) // 6  # Cada obstáculo tiene 6 valores

        if num_obstacles == 0:
            rospy.logwarn("No se detectaron obstaculos.")
            return  # No hay obstáculos detectados

        for i in range(num_obstacles):
            width = msg.data[6 * i + 4]  # Ancho del obstáculo
            height = msg.data[6 * i + 5]  # Alto del obstáculo

            # Mantener el máximo tamaño detectado
            if width > self.max_obstacle_width:
                self.max_obstacle_width = width
            if height > self.max_obstacle_height:
                self.max_obstacle_height = height

    def calculate_average_distance(self):
        """
        Calcula la distancia promedio entre puntos consecutivos en la trayectoria,
        personalizando según el tipo de trayectoria.
        """
        if len(self.goalx) < 2:  # Evita errores si la trayectoria no tiene suficientes puntos
            print("No hay suficientes puntos en la trayectoria", self.trajectory_name, "para calcular distancia promedio.")
            return 0.0, 0.0

        if self.trajectory_name == "rectangle":
            # 📌 En rectángulo, calculamos distancias separadas para X e Y
            avg_distance_x, avg_distance_y = self.calculate_rectangle_distances()
            print("Distancia promedio para", self.trajectory_name,":", "X = ",avg_distance_x, "Y = ",avg_distance_y)
            return avg_distance_x, avg_distance_y

        # Para otras trayectorias (Elipse, Lemniscata)
        distances = np.sqrt(np.diff(self.goalx) ** 2 + np.diff(self.goaly) ** 2)
        avg_distance = np.mean(distances) if len(distances) > 0 else 0.0

        # 📌 Para la elipse, usamos la fórmula de Ramanujan para la longitud de la curva
        if self.trajectory_name == "ellipse":
            circumference = np.pi * (3 * (self.params[0] + self.params[1]) - np.sqrt((3 * self.params[0] + self.params[1]) * (self.params[0] + 3 * self.params[1])))
            avg_distance = circumference / len(self.goalx)

        # 📌 Para la lemniscata, usamos una aproximación del recorrido total
        elif self.trajectory_name == "lemniscate":
            # 📌 En la lemniscata, el espaciamiento de los puntos no es uniforme
            distances = np.sqrt(np.diff(self.goalx) ** 2 + np.diff(self.goaly) ** 2)

            if len(distances) > 0:
                avg_distance = np.mean(distances)  # Usamos la distancia promedio real de los puntos
            else:
                avg_distance = 0.0

            print("Distancia promedio corregida para lemniscate:", avg_distance)
            return avg_distance, avg_distance

        print("Distancia promedio para",self.trajectory_name, avg_distance, "m")
        return avg_distance, avg_distance  # Retornamos valores iguales para ejes X e Y

    def calculate_rectangle_distances(self):
        """
        Calcula la distancia promedio entre puntos en X y Y de una trayectoria rectangular.
        """
        length = self.params[0]  # Largo del rectángulo
        width = self.params[1]   # Ancho del rectángulo
        num_points = len(self.goalx)

        if num_points < 4:
            rospy.logwarn("No hay suficientes puntos en la trayectoria rectangular.")
            return 0.0, 0.0

        num_points_per_side = num_points // 4  # Se divide en los 4 lados
        avg_distance_x = length / num_points_per_side
        avg_distance_y = width / num_points_per_side

        return avg_distance_x, avg_distance_y

    def calculate_future_index(self, obstacle_size):
        """
        Calcula cuántos índices adelante hay que moverse para evadir el obstáculo.
        :param obstacle_size: Tamaño del obstáculo (ancho o alto).
        :return: Número de índices que deben saltarse en la trayectoria.
        """
        avg_distance_x, avg_distance_y = self.calculate_average_distance()

        if self.trajectory_name == "rectangle":
            # 📌 Decidir si moverse en X o en Y según la dirección de la evasión
            if abs(self.goalx[self.evasion_start_index + 1] - self.goalx[self.evasion_start_index]) > \
               abs(self.goaly[self.evasion_start_index + 1] - self.goaly[self.evasion_start_index]):
                steps_needed = int(np.ceil((obstacle_size + 0.25) / avg_distance_x)) if avg_distance_x > 0 else 10
            else:
                steps_needed = int(np.ceil((obstacle_size + 0.25) / avg_distance_y)) if avg_distance_y > 0 else 10
        else:
            steps_needed = int(np.ceil((obstacle_size + 0.25) / avg_distance_x)) if avg_distance_x > 0 else 10

        print("Necesarios", steps_needed, "indices para evadir el obstaculo")
        return steps_needed


    def find_future_goal(self, current_index):
        """
        Encuentra un punto de evasión basado en la distancia entre puntos de la trayectoria.
        """
        if self.evasion_start_index is None:
            self.evasion_start_index = current_index
            print("Comenzando evasion en indice", self.evasion_start_index)

        # 📌 Determinar la cantidad de índices que deben moverse
        obstacle_size = max(self.max_obstacle_width, self.max_obstacle_height)
        future_steps = self.calculate_future_index(obstacle_size)

        # 📌 Seleccionar el índice futuro
        future_index = min(self.evasion_start_index + future_steps, len(self.goalx) - 1)
        goal_x, goal_y = self.goalx[future_index], self.goaly[future_index]

        print("Nuevo punto de evasion encontrado:", goal_x, goal_y, "indice:", future_index)
        return goal_x, goal_y, future_index

    def calculate_attractive_force(self, x, y, evasion_goal):
        goal_x, goal_y = evasion_goal  # Usar el punto de evasión seleccionado en run()

        goal_vector = np.array([goal_x - x, goal_y - y])
        distance_to_goal = np.linalg.norm(goal_vector)

        if distance_to_goal < self.dstar:
            attractive_force = -self.zeta * goal_vector
        else:
            attractive_force = -(self.dstar / (distance_to_goal + 1e-6)) * self.zeta * goal_vector

        return attractive_force, distance_to_goal 

    def calculate_repulsive_force(self, x, y):
        repulsive_force = np.array([0.0, 0.0])
        epsilon = 1e-6

        if self.hough_lines:
            for (x1, y1, x2, y2) in self.hough_lines:
                obstacle_vector = np.array([x - x1, y - y1])
                distance_to_obstacle = np.linalg.norm(obstacle_vector)
                if distance_to_obstacle > epsilon and distance_to_obstacle < self.Qstar:
                    direction = obstacle_vector / distance_to_obstacle
                    repulsion_strength = self.eta * (1 / self.Qstar - 1 / distance_to_obstacle) * (1 / distance_to_obstacle**2)
                    repulsive_force += repulsion_strength * direction

        return repulsive_force
        
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
        J = np.array([[r2 * np.cos(th1), r2 * np.sin(th1), r2 * np.cos(th1), r2 * np.sin(th1)],
        		      [r2 * np.sin(th1), -r2 * np.cos(th1), r2 * np.sin(th1), -r2 * np.cos(th1)],
                      [-1 / (self.lx + self.ly), -1 / (self.lx + self.ly), 1 / (self.lx + self.ly), 1 / (self.lx + self.ly)]])

        J_2 = np.array([[r2 * np.cos(th1), r2 * np.sin(th1), r2 * np.cos(th1), r2 * np.sin(th1)],
        		        [r2 * np.sin(th1), -r2 * np.cos(th1), r2 * np.sin(th1), -r2 * np.cos(th1)]])
        		 
        J_2 = (self.r / 4) * J_2
        J = (self.r / 4) * J
        J_inv = np.linalg.pinv(J)
        J_2_inv = np.linalg.pinv(J_2)
        
        return J_inv, J_2, J_2_inv

    def get_inv_Jacobian_APF(self,th):
        th1 = th + np.pi/4
        r2 = np.sqrt(2)
        J_inv_APF = np.array([[r2 * np.cos(th1) , r2 * np.sin(th1), -(self.lx + self.ly)],
                          [r2 * np.sin(th1) ,-r2 * np.cos(th1), -(self.lx + self.ly)],
                          [r2 * np.cos(th1) , r2 * np.sin(th1),  (self.lx + self.ly)],
                          [r2 * np.sin(th1) ,-r2 * np.cos(th1),  (self.lx + self.ly)]])
        return J_inv_APF
    
    def run(self):
    
        init_time = rospy.Time.now()
        last_time = init_time
        
        for i in range(0,len(self.goalx)):
                 
            while not rospy.is_shutdown() and (rospy.Time.now()-init_time).to_sec() < self.time[i]:
                pass
            #print(1)    
            dt = (rospy.Time.now()-last_time).to_sec()
            last_time = rospy.Time.now()
            
            # Errores
            e_x = self.goalx[i] - self.x
            e_y = self.goaly[i] - self.y
            e_theta = self.goaltheta[i] - self.theta
            e_theta = (e_theta + np.pi) % (2*np.pi) - np.pi
            
            # Leyes de control de las tareas
            acx_task1 = (self.Kp * np.tanh(e_x) + self.goalx_d[i])
            acy_task1 = (self.Kp * np.tanh(e_y) + self.goaly_d[i])
            acw_task2 = (self.Kp * np.tanh(e_theta) + self.goaltheta_d[i])
            
            # Definir vectores
            ac_task1 = np.array([[acx_task1], [acy_task1],[0]])
            ac_task2 = np.array([[0],[0],[acw_task2]])

            repulsive_force = self.calculate_repulsive_force(self.x, self.y)

            if np.any(repulsive_force != 0):  # Se detecta obstáculo
                if not self.using_apf:  # Si es la primera vez que detectamos el obstáculo
                    self.using_apf = True
                    self.evasion_start_index = i
                    self.fixed_index = i 
                    print("Comenzando evasion en indice", self.evasion_start_index)

                    # Buscar un punto que cubra al obstáculo detectado
                    new_evasion_goal_x, new_evasion_goal_y, new_evasion_index = self.find_future_goal(self.evasion_start_index)
                    
                    print("Punto de evasion encontrado:", new_evasion_goal_x, new_evasion_goal_y, "indice:",new_evasion_index)
                    self.evasion_goal = (new_evasion_goal_x, new_evasion_goal_y)

                # Si el obstáculo "crece", actualizar el punto de evasión
                elif self.max_obstacle_width > 0 and self.max_obstacle_height > 0:
                    new_evasion_goal_x, new_evasion_goal_y, new_evasion_index = self.find_future_goal(self.evasion_start_index)

                    # 📌 Solo actualizar si es un punto nuevo
                    if (new_evasion_goal_x, new_evasion_goal_y) != self.evasion_goal and new_evasion_index > i:
                        print("Ajustando punto de evasion:", new_evasion_goal_x, new_evasion_goal_y, "indice:", new_evasion_index)
                        self.evasion_goal = (new_evasion_goal_x, new_evasion_goal_y)

            if self.using_apf:  # Si APF ya fue activado, permanece en APF hasta alcanzar el objetivo
                
                attractive_force, distance_to_goal = self.calculate_attractive_force(self.x, self.y, self.evasion_goal)
                total_force = attractive_force + repulsive_force
                
                # Cálculo de referencia de orientación
                theta_ref = np.arctan2(-total_force[1], -total_force[0])
                error_theta = theta_ref - self.theta
                error_theta = (error_theta + np.pi) % (2 * np.pi) - np.pi

                # Control de velocidad angular y lineal basado en APF
                acw = np.clip(self.kp_theta * error_theta, -self.w_max, self.w_max)
                v_ref = min(np.linalg.norm(-total_force), self.v_max)
                acx = v_ref * np.cos(self.theta) * 0.8
                acy = v_ref * np.sin(self.theta)

                u = np.array([[acx],[acy],[acw]])
                J_inv_APF = self.get_inv_Jacobian_APF(self.theta)
                w = np.dot(J_inv_APF,u)/self.r

                # Condición para desactivar APF y volver a PI
                if distance_to_goal <= self.position_accuracy:
                    print("Objetivo alcanzado, cambiando a control PI.")

                    # Buscar índice para reanudar
                    min_dist = float('inf')
                    closest_index = -1

                    for k in range(self.fixed_index + 20, len(self.goalx)):
                        dist = math.sqrt((self.x - self.goalx[k])**2 + (self.y - self.goaly[k])**2)
                        if dist < min_dist:
                            min_dist = dist
                            closest_index = k

                    if min_dist < 0.05:
                        print("Objetivo de evasión alcanzado, volviendo a control PI.")
                        print("Índice trayectoria más cercano:", closest_index)
                        i = closest_index
                        self.aux_apf = closest_index

                        #Resetear la memoria del obstáculo aquí
                        self.max_obstacle_width = 0.0
                        self.max_obstacle_height = 0.0
                        self.using_apf = False  # Reinicia para permitir PI en la siguiente ejecución
                        self.evasion_goal = None  # Resetear el punto de evasión


                        self.using_apf = False
                        self.evasion_goal = None
                        self.evasion_start_index = None

                        init_time = rospy.Time.now() - rospy.Duration.from_sec(self.time[i])

                        # Esperar que el tiempo actual alcance el tiempo del nuevo índice
                        while not rospy.is_shutdown() and (rospy.Time.now() - init_time).to_sec() < self.time[i]:
                            pass

            else:

                J_inv, J_2, J_inv_2 = self.get_inv_Jacobian(self.theta)

                # Null-space projector for Task 1 (3x3 matrix)
                I_3 = np.eye(4)
                N_task1 = I_3 - np.dot(J_inv_2,J_2)
                w = np.dot(J_inv,ac_task1) + np.dot(np.dot(N_task1,J_inv),ac_task2)

            
            w1_aux = w[0,0]
            w2_aux = w[1,0]
            w3_aux = w[2,0]
            w4_aux = w[3,0]
            a = 9.00
            w1 = max(min(w1_aux, a), -a)
            w2 = max(min(w2_aux, a), -a)
            w3 = max(min(w3_aux, a), -a)
            w4 = max(min(w4_aux, a), -a)
            
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
