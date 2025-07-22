#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import rospy
import rospkg
import matplotlib.pyplot as plt
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
import multiprocessing as mp


class PoseControl:
    def __init__(self):
        rospy.init_node("trajectory_controller", anonymous=False)
        # Load and build the fuzzy logic system from the JSON file
        self.engine_vx = self.build_fuzzy_engine("/home/jetauto/JetAuto_VA_ws/src/jetauto_trajectory_control/src/simple_control_TRI.json")
        self.engine_vy = self.build_fuzzy_engine("/home/jetauto/JetAuto_VA_ws/src/jetauto_trajectory_control/src/simple_control_TRI.json")
        self.engine_w = self.build_fuzzy_engine("/home/jetauto/JetAuto_VA_ws/src/jetauto_trajectory_control/src/simple_control_TRI.json")
        
        path_type = rospy.get_param('path_type', 'ellipse')
        self.tm = rospy.get_param('tiempo_muestreo', 0.1)
        self.tf = rospy.get_param('tiempo_total', 80)
        self.r = rospy.get_param('r', 0.0485)
        self.lx = rospy.get_param('lx', 0.0975)
        self.ly = rospy.get_param('ly', 0.103)
        self.guardar_datos = rospy.get_param('guardar_datos', True)
        
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
        
        #Constantes de control para Fuzzy
        self.K1 = rospy.get_param('Fuzzy/K1', 0.1)
        self.K2 = rospy.get_param('Fuzzy/K2', 0.1)
        self.Kf = rospy.get_param('Fuzzy/Kf', 0.01)
        
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
        
        self.svx_ant = 0.0
        self.svy_ant = 0.0
        self.sw_ant = 0.0

        self.s_vx = 0.0
        self.s_vy = 0.0
        self.s_w = 0.0

        self.sp_vx = 0.0
        self.sp_vy = 0.0
        self.sp_w = 0.0
        
        self.fvx = 0.0
        self.fvy = 0.0
        self.fw = 0.0
       
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
            directory = os.path.join(package_path,'datos','SMC')
            if not os.path.exists(directory):
                os.makedirs(directory)  # Create the directory if it doesnt exist
            self.file_name = os.path.join(directory, "FSMC_Mandami_{}.txt".format(path_type))
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
        
    def build_fuzzy_engine(self,json_path):
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

        # Manejo de entradas
        for input_var in data["Inputs"]:
            iv = fl.InputVariable(
                name=input_var["Name"],
                minimum=input_var["Range"][0],
                maximum=input_var["Range"][1],
                lock_range=False,
                terms=[mf_type_mapping[mf["Type"].lower()](mf["Name"], mf["Parameters"]) for mf in input_var["MembershipFunctions"]],
            )
            engine.input_variables.append(iv)

        # Manejo de salida única
        output_var = data["Outputs"]
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

        # Manejo de reglas
        rb = fl.RuleBlock(
            name="rules",
            conjunction=fl.Minimum(),
            disjunction=None,
            implication=fl.Minimum(),
            activation=fl.General(),
        )

        for rule in data["Rules"]:
            # Generar la parte del antecedente
            antecedent = " and ".join(
                f"{data['Inputs'][i]['Name']} is {data['Inputs'][i]['MembershipFunctions'][ante - 1]['Name']}"
                for i, ante in enumerate(rule["Antecedent"])
            )

            # Generar la parte del consecuente
            consequent = f"{output_var['Name']} is {output_var['MembershipFunctions'][rule['Consequent'] - 1]['Name']}"

            # Crear y agregar la regla al bloque
            rb.rules.append(fl.Rule.create(f"if {antecedent} then {consequent}", engine))

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
        plt.plot(self.t, self.w1_ref,label='w1')
        plt.plot(self.t,self.w2_ref,label='w2')
        plt.plot(self.t, self.w3_ref,label='w3')
        plt.plot(self.t,self.w4_ref,label='w4')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Velocidad ruedas')
        plt.grid(True)
        plt.legend() 
        plt.show()
    
    def get_inv_Jacobian(self,th):
        th1 = th + np.pi/4
        r2 = np.sqrt(2)
        J_inv = np.array([[r2 * np.cos(th1) , r2 * np.sin(th1), -(self.lx + self.ly)],
                          [r2 * np.sin(th1) ,-r2 * np.cos(th1), -(self.lx + self.ly)],
                          [r2 * np.cos(th1) , r2 * np.sin(th1),  (self.lx + self.ly)],
                          [r2 * np.sin(th1) ,-r2 * np.cos(th1),  (self.lx + self.ly)]])
        return J_inv

    def fuzzy_vx(self):
        self.engine_vx.input_variable("s").value = self.K1*self.s_vx
        self.engine_vx.input_variable("sp").value = self.K2*self.sp_vx
        self.engine_vx.process()
        self.fvx = self.engine_vx.output_variable("u").value


    def fuzzy_vy(self):
        self.engine_vy.input_variable("s").value = self.K1*self.s_vy
        self.engine_vy.input_variable("sp").value = self.K2*self.sp_vy
        self.engine_vy.process()
        self.fvy = self.engine_vy.output_variable("u").value


    def fuzzy_w(self):
        self.engine_w.input_variable("s").value = self.K1*self.s_w
        self.engine_w.input_variable("sp").value = self.K2*self.sp_w
        self.engine_w.process()
        self.fw = self.engine_w.output_variable("u").value

    
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
            
            #Superficie deslizante PID de velocidad lineal
            svx = self.lambda_1v*e_x + self.lambda_0*ei_x + ed_x
            self.s_vx = svx / (np.abs(svx) + self.delta)
            self.sp_vx = (self.s_vx - self.svx_ant)/self.tm
            
            svy = self.lambda_1v*e_y + self.lambda_0*ei_y + ed_y
            self.s_vy = svy / (np.abs(svy) + self.delta)
            self.sp_vy = (self.s_vy - self.svy_ant)/self.tm
            
            #Superficie deslizante PID de velocidad angular
            sw = self.lambda_1v*e_theta + self.lambda_0*ei_theta + ed_theta
            self.s_w = sw / (np.abs(sw) + self.delta)
            self.sp_w = (self.s_w - self.sw_ant)/self.tm
            
            # Acciones de control continua
            vx_ac = self.tau_v*(vx*(1/(self.tau_v) - self.lambda_1v) + self.lambda_1v * self.goalx_d[i] + self.lambda_0 * e_x)
            vy_ac = self.tau_v*(vy*(1/(self.tau_v) - self.lambda_1v) + self.lambda_1v * self.goaly_d[i] + self.lambda_0 * e_y)
            w_ac =  self.tau_w*(w*(1/(self.tau_w) - self.lambda_1w) + self.lambda_1w * self.goaltheta_d[i] + self.lambda_0 * e_theta)
            tic = time.time()
            # Acciones de control Fuzzy
            p1 = mp.Process(target=self.fuzzy_vx())
            p2 = mp.Process(target=self.fuzzy_vy())
            p3 = mp.Process(target=self.fuzzy_w())

            # Iniciar los procesos
            p1.start()
            p2.start()
            p3.start()

            # Esperar a que terminen
            p1.join()
            p2.join()
            p3.join()
            toc = time.time()
            print("Ejecucion ", toc-tic)
            # Accion de control general
            ac_vector = np.array([[vx_ac],[vy_ac],[w_ac]]) + self.Kf * np.array([[self.fvx],[self.fvy],[self.fw]])
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
            
            # Datos i-1 superficies
            self.svx_ant = self.s_vx
            self.svy_ant = self.s_vy
            self.sw_ant  = self.s_w

            #print(w1)
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
