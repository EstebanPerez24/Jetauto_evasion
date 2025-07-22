#!/usr/bin/env python
# -*- coding:utf-8 -*-
#para el read encoder
import os
import smbus2
import rospy
import math
import sys
import time
import threading
from jetauto_interfaces.msg import vel_wheels  #ros messages
from jetauto_interfaces.msg import imu_encoder   #ros messages
# para la IMU
import serial
import struct
import platform
import serial.tools.list_ports
import tf
from sensor_msgs.msg import Imu
from sensor_msgs.msg import MagneticField
from sensor_msgs.msg import NavSatFix
from tf.transformations import quaternion_from_euler
from math import atan2, degrees
from std_msgs.msg import Float64
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
import numpy as np


#  --------------------------------------------------------Para el read Encoder --------------------------------------------------------
ENCODER_MOTOR_MODULE_ADDRESS = 0x34
                      
imu_encoder_msg = imu_encoder()                 # Mensaje con IMU y Velocidades de cada rueda
mag_msg = MagneticField()


class Driver_Sensors:
    def __init__(self, i2c_port = 1, pulse_per_cycle = 3960, motor_type=3):
        rospy.init_node('Driver_Sensors')
        self.i2c_port = i2c_port
        self.machine_type = os.environ.get('MACHINE_TYPE')
        self.current_time_all = rospy.Time.now()		# tiempos para calcular velocidad
        self.previous_time_all = rospy.Time.now()
        self.clear_encoder()												# limpiar el encoder
        self.pulse_per_cycle = pulse_per_cycle										# pulsos por vuelta del encoder 
        self.vel_prev = [0,0,0,0]
        self.pulsos_prev = [0,0,0,0]
        self.pulsos_aux = [0,0,0,0]
        self.time_prev = time.time()											# memoria de velocidad
        self.v = [0, 0, 0, 0]
        self.stamp = rospy.get_rostime()
#------------------ Para el filtrado --------------------
        self.encoder_values = [0, 0, 0, 0]
        self.previous_encoder_values = [0, 0, 0, 0]
        self.previous_time = rospy.get_rostime()
        self.pulse_counts = [0, 0, 0, 0]
        self.angular_speeds = [0.0, 0.0, 0.0, 0.0]
        self.filtered_angular_speeds = [0.0, 0.0, 0.0, 0.0]
        self.filtered_angular_speeds_prev = [0.0, 0.0, 0.0, 0.0]
        self.n = 0  # Number of pulses to wait before computing speed
        self.max_delta_threshold = 600  # Threshold for sudden changes
        self.alpha = 0.20
#-------------------Fin Filtrado -------------------------
        #self.file = open("/home/jetauto/JetAuto_VA_ws/src/Datos_Graficas/Vel_Filtered_Unfiltered3.txt","w")
        #self.file.write("t\tw1f\tw2f\tw3f\tw4f\tw1\tw2\tw3\tw4\n")
        #self.file.close()
        #self.t1 = self.stamp
        

        """with smbus2.SMBus(self.i2c_port) as bus:
            bus.write_i2c_block_data(ENCODER_MOTOR_MODULE_ADDRESS, 20, [motor_type, ])					# configuracion de tipo de motor, tipo 3 para 	motor con encoder"""

    def set_speed(self, speed, motor_id=None, offset=0):				# configura la velocidad de las ruedas
        w_write = [0,0,0,0]
        with smbus2.SMBus(self.i2c_port) as bus:
            # motor speed  control register address is 51 + motor_id - 1
            if motor_id is None:							# en caso de no especificar el motor, en speed se lee un arreglo de 4
                for id_index in range(len(speed)):					# nuevo id para escribir en los registros del modulo
                    new_id = 1
                    i = id_index
                    if id_index == 0:
                        new_id = 2
                    elif id_index == 2:
                        new_id = 3
                    elif id_index == 1:
                        new_id = 1
                    elif id_index == 3:
                        new_id = 4
                    motor_speed = speed[id_index]
                    sp = motor_speed
                    if sp > 100:							# limitar el setpoint de velocidad entre -100 y 100
                        sp = 100
                    elif sp < -100:
                        sp = -100
                    try:
                        bus.write_i2c_block_data(ENCODER_MOTOR_MODULE_ADDRESS, 50 + new_id, [sp, ])		# escribir a partir del registro 51 la velocidad de las rueda
                    except BaseException as e:
                        print(e)
                        bus.write_i2c_block_data(ENCODER_MOTOR_MODULE_ADDRESS, 50 + new_id, [sp, ])

            else:									# en caso de escribir un motor id al llamar la funcion, speed es un entero
                if 0 < motor_id <= 4:
                    if motor_id == 0:							# nuevo id para escribir en los registros del modulo 
                        motor_id = 2
                    elif motor_id == 2:
                        motor_id = 3
                    elif motor_id == 1:
                        motor_id = 1
                    elif motor_id == 3:
                        motor_id = 4
                    speed = 100 if speed > 100 else speed
                    speed = -100 if speed < -100 else speed
                    try:
                        bus.write_i2c_block_data(ENCODER_MOTOR_MODULE_ADDRESS, 50 + motor_id, [speed, ])
                        
                    except BaseException as e:
                        print(e)
                        bus.write_i2c_block_data(ENCODER_MOTOR_MODULE_ADDRESS, 50 + motor_id, [speed, ])
                        
                else:
                    raise ValueError("Invalid motor id")				# en caso de ingresar un id de motor fuera de rango


    def clear_encoder(self, motor_id=None):									# escribe 0 en el registro de conteo de pulsos
        with smbus2.SMBus(self.i2c_port) as bus:
            if motor_id is None:										# en caso de no especificar el motor, se reseatea todos los registros
                self.previous_time_all = rospy.Time.now() # referencia de tiempo tras limpiar el encoder
                bus.write_i2c_block_data(ENCODER_MOTOR_MODULE_ADDRESS, 60, [0]*16)				# desde el registro 60 (cada rueda utiliza 4 bytes)
            else:
                if 0 <= motor_id < 4:										# en caso de especificar el id motor, se limpia el registro respectivo
                    self.previous_time_all = rospy.Time.now()
                    bus.write_i2c_block_data(ENCODER_MOTOR_MODULE_ADDRESS, 60 + motor_id*4, [0]*4)
                else:
                    raise ValueError("Invalid motor id")							# en caso de ingresar un id motor invalido

    def read_encoder(self, motor_id=None):									# lee el numero de pulsos acumulado en los encoders
        
        with smbus2.SMBus(self.i2c_port) as bus:
            if 0 <= motor_id < 4:										# en caso de ingresar un motor id especifico
                i = motor_id
                w = [0,0,0,0]
                data = bus.read_i2c_block_data(ENCODER_MOTOR_MODULE_ADDRESS, 60, 16)        # Lectura de los pulsos de las 4 ruedas
                self.current_time_all = rospy.Time.now()
                w1 = data[3] << 24 | data[2] << 16 | data[1] << 8 | data[0]                 # Ordenar los bytes de cada llanta 
                if data[3] >= 255:
                    w1 = w1-4294967295

                w2 = data[7] << 24 | data[6] << 16 | data[5] << 8 | data[4]
                if data[7] >= 255:
                    w2 = w2-4294967295

                w3 = data[11] << 24 | data[10] << 16 | data[9] << 8 | data[8]
                if data[11] >= 255:
                    w3 = w3-4294967295

                w4 = data[15] << 24 | data[14] << 16 | data[13] << 8 | data[12]
                if data[15] >= 255:
                    w4 = w4-4294967295

                w = [w1, w2, w3, w4]
                return w
            else:
                raise ValueError("Invalid motor id")
            

    def compute_speed(self):									# calcula la velocidad de los motores a partir de los pulsos       
        
        pulsos = [0, 0, 0, 0]
        i = 0
        aux_filtered = rospy.get_param("enable_ang_speed_filter", True);        #en caso true halla velocidad filtrada, sino no aplica el filtro
        self.pulsos_aux = self.read_encoder(i)
        dt = (self.current_time_all-self.previous_time_all).to_sec()
        self.previous_time_all = self.current_time_all
        
        #hola = open("/home/jetauto/JetAuto_VA_ws/src/Datos_Graficas/Vel_Filtered_Unfiltered3.txt","a")
        if aux_filtered:
            
            self.v = self.compute_speed_filtered(dt)                              # Halla velocidad filtrada a partir de pulsos
            #self.file.write(w1+"\t")
            #hola.write(str((self.stamp-self.t1).to_sec())+"\t")
            #hola.write(str(-go[1])+"\t"+str(go[0])+"\t"+str(-go[2])+"\t"+str(go[3])+"\t")
        else:                                                                   # Caso velocidad sin filtro
            for i in range(4):
                pulsos[i] = self.pulsos_aux[i] - self.pulsos_prev[i]
                self.pulsos_prev[i] = self.pulsos_aux[i]          
                self.v[i] = pulsos[i]*2.0*math.pi/(self.pulse_per_cycle*dt)					# caculo de la velocidad a partir de numero de pulsos, en rad/s
                error = abs(self.vel_prev[i]-self.v[i])
                if error > 50:										# correcion en caso de medida anormal
                    self.v[i] = self.vel_prev[i]
                self.vel_prev[i] = self.v[i]
                #print("bien")
                
                #print("bien455")
                
            #hola.write(str(-self.v[1])+"\t"+str(self.v[0])+"\t"+str(-self.v[2])+"\t"+str(self.v[3])+"\n")
            #hola.close()
        return self.v
        
    def compute_speed_filtered(self, dt):
        current_time = self.stamp				# Cambiar por el rospy.time.now()
        self.encoder_values = [self.pulsos_aux[0], self.pulsos_aux[1], self.pulsos_aux[2], self.pulsos_aux[3]]    
        delta_time = dt
        if delta_time > 0.005:
            for i in range(4):
                delta_encoder = self.encoder_values[i] - self.previous_encoder_values[i]

                # Check for sudden large changes
                if abs(delta_encoder) > self.max_delta_threshold:
                    #rospy.logwarn("Sudden change detected on encoder {i}: {delta_encoder}")
                    self.previous_encoder_values[i] = self.encoder_values[i]
                    self.pulse_counts[i] = 0
                    continue

                if delta_encoder != 0:
                    self.pulse_counts[i] += abs(delta_encoder)

                if self.pulse_counts[i] >= self.n:
                    self.angular_speeds[i] = delta_encoder * 2 * math.pi/ (delta_time*3960)
                    self.pulse_counts[i] = 0  # Reset pulse count after computing speed
            
            self.apply_low_pass_filter()
            
            #v = 
            #if any(speed != 0 for speed in self.filtered_angular_speeds):  # Check if there is any non-zero speed
            #self.speed_pub.publish(Float32MultiArray(data=[-self.filtered_angular_speeds[0],self.filtered_angular_speeds[1],-self.filtered_angular_speeds[2],self.filtered_angular_speeds[3]]))
            #print(i, self.filtered_angular_speeds[i])
            
            self.previous_encoder_values = self.encoder_values[:]
            self.previous_time = current_time
        return self.filtered_angular_speeds

        

    def apply_low_pass_filter(self):
        for i in range(4):
            self.filtered_angular_speeds[i] = self.alpha * self.angular_speeds[i] + (1 - self.alpha) * self.filtered_angular_speeds[i]
            
          #  if self.filtered_angular_speeds[i] > 20:
         #       self.filtered_angular_speeds[i] = self.filtered_angular_speeds_prev[i]
        #self.filtered_angular_speeds_prev = self.filtered_angular_speeds

    def publish_speed(self, event = None):									# publica la velocidad calculada de cada llanta
        self.stamp = rospy.get_rostime()                            # Stamp para publicar el menasaje
        w = [0, 0, 0, 0]
        w = self.compute_speed()                                    # Calcula la velocidad de cada llanta
        [imu_encoder_msg.w2, imu_encoder_msg.w1, imu_encoder_msg.w3, imu_encoder_msg.w4] =[w[0],-w[1],-w[2],w[3]]		# Colocar la velocidad en el mensaje
        imu_encoder_msg.imu.header.stamp = self.stamp                           # colocar el Stamp en el Mensaje
        imu_encoder_pub.publish(imu_encoder_msg)
        mag_pub.publish(mag_msg)
        #print("hola")

#  ---------------------------------------------------------------------------------------------------------------------------

#  --------------------------------------------------------Para la IMU --------------------------------------------------------

def find_ttyUSB():
    print('The default serial port of the imu is /dev/ttyUSB0, if multiple serial port devices are identified, modify the serial port corresponding to the imu in the launch file')
    posts = [port.device for port in serial.tools.list_ports.comports() if 'USB' in port.device]
    print('There are {} {} serial port devices connected to the current PC: {}'.format(len(posts), 'USB', posts))


def checkSum(list_data, check_data):
    return sum(list_data) & 0xff == check_data


def hex_to_short(raw_data):
    return list(struct.unpack("hhhh", bytearray(raw_data)))


def hex_to_data(raw_data):
    return list(struct.unpack("i", bytearray(raw_data)))
    

def hex_to_altitude(raw_data):
    return list(struct.unpack("h", bytearray(raw_data)))
    

def handleSerialData(raw_data):
    global buff, key, angle_degree, magnetometer, acceleration, angularVelocity, pub_flag, readreg, calibuff, flag, mag_offset, mag_range, version, longitude_imu, latitude_imu, altitude_imu, offset_mag, theta, time_ant, angular_vel_ant
    angle_flag=False
    if python_version == '2':
        buff[key] = ord(raw_data)
    if python_version == '3':
        buff[key] = raw_data

    key += 1
    if buff[0] != 0x55:
        key = 0
        return
    if key < 11:  
        return
    else:
        data_buff = list(buff.values()) 
        if buff[1] == 0x51 :
            if checkSum(data_buff[0:10], data_buff[10]):
                acceleration = [hex_to_short(data_buff[2:10])[i] / 32768.0 * 16 * 9.8 for i in range(0, 3)]
            else:
                print('0x51 Check failure')

        elif buff[1] == 0x52:
            if checkSum(data_buff[0:10], data_buff[10]):
                angularVelocity = [hex_to_short(data_buff[2:10])[i] / 32768.0 * 2000 * math.pi / 180 for i in range(0, 3)]
                #print(angularVelocity)

            else:
                print('0x52 Check failure')

        elif buff[1] == 0x53:
            if checkSum(data_buff[0:10], data_buff[10]):
                temp = hex_to_short(data_buff[2:10])
                angle_degree = [temp[i] / 32768.0 * 180 for i in range(0, 3)]
                version = temp[3]
                angle_flag = True
            else:
                print('0x53 Check failure')
        elif buff[1] == 0x54:
            if checkSum(data_buff[0:10], data_buff[10]): 
                magnetometer = hex_to_short(data_buff[2:10])
                if flag:
                    calibuff.append(magnetometer[0:2])
            else:
                print('0x54 Check failure')


        elif buff[1] == 0x57:
            if checkSum(data_buff[0:10], data_buff[10]):
                longitude_imu = (hex_to_data(data_buff[2:6])[0]  // 10000000.0 * 100 ) +  ((hex_to_data(data_buff[2:6])[0]  % 10000000) / 10000000.0)
                latitude_imu = (hex_to_data(data_buff[6:10])[0]  // 10000000.0 * 100 ) +((hex_to_data(data_buff[6:10])[0] % 10000000) / 10000000.0)
            else:
                print('0x57 Check failure')
                
                
                
        elif buff[1] == 0x58:
            if checkSum(data_buff[0:10], data_buff[10]): 
                altitude_imu = hex_to_altitude(data_buff[2:4])[0]  / 10.0
                
            else:
                print('0x58 Check failure')
                

        elif buff[1] == 0x5f:
            if checkSum(data_buff[0:10], data_buff[10]):
                readval = hex_to_short(data_buff[2:10])
                if readreg == 0x0b:
                    mag_offset = readval
                else:
                    mag_range = readval

                print(readval)
            else:
                print('0x5f Check failure')

        else:
            #print("该数据处理类没有提供该 " + str(buff[1]) + " 的解析")
            #print("或数据错误")
            buff = {}
            key = 0

        buff = {}
        key = 0
        if angle_flag:
            stamp = rospy.get_rostime()

            imu_msg.header.stamp = stamp
            imu_msg.header.frame_id = "base_link"

            mag_msg.header.stamp = stamp
            mag_msg.header.frame_id = "base_link"

            location_msg.header.stamp = stamp
            location_msg.header.frame_id = "base_link"
            
            angle_radian = [angle_degree[i] * math.pi / 180 for i in range(3)]
            qua = quaternion_from_euler(angle_radian[0], angle_radian[1], angle_radian[2])
            
            (r,p,y) = tf.transformations.euler_from_quaternion((qua[0],qua[1],qua[2],qua[3]))
            y = (y - offset_mag)
            imu_encoder_msg.angle = (y + np.pi) % (2*np.pi) - np.pi			#Angulo en radianes
            #y = math.radians(y)
            
            Mx = magnetometer[0]*math.cos(p) + magnetometer[2]*math.sin(p)
            My = magnetometer[0]*math.sin(r)*math.sin(p) + magnetometer[1]*math.cos(r) - magnetometer[2]*math.sin(r)*math.cos(p) 
            
            angle_s = -math.atan2(My,Mx) - offset_mag2
            angle_s = (angle_s + np.pi) % (2*np.pi) - np.pi
            #print(angle_s)
            
            qua = quaternion_from_euler(r, p, y)    
       
            theta = theta + (angularVelocity[2] + angular_vel_ant)* 0.5* (rospy.Time.now() - time_ant).to_sec()
            imu_encoder_msg.angle = (theta + np.pi) % (2*np.pi) - np.pi			#Angulo en radianes
            time_ant = rospy.Time.now()
            angular_vel_ant = angularVelocity[2]
            print(theta)
            
            imu_encoder_msg.imu.orientation.x = qua[0]
            imu_encoder_msg.imu.orientation.y = qua[1]
            imu_encoder_msg.imu.orientation.z = qua[2]
            imu_encoder_msg.imu.orientation.w = qua[3]
            #print(imu_encoder_msg.imu.orientation)

            imu_encoder_msg.imu.angular_velocity.x = angularVelocity[0]
            imu_encoder_msg.imu.angular_velocity.y = angularVelocity[1]
            imu_encoder_msg.imu.angular_velocity.z = angularVelocity[2]

            imu_encoder_msg.imu.linear_acceleration.x = acceleration[0]
            imu_encoder_msg.imu.linear_acceleration.y = acceleration[1]
            imu_encoder_msg.imu.linear_acceleration.z = acceleration[2]
            mag_msg.magnetic_field.x = magnetometer[0]
            mag_msg.magnetic_field.y = magnetometer[1]
            mag_msg.magnetic_field.z = magnetometer[2]
            #imu_encoder_pub.publish(imu_encoder_msg)

altitude_imu = 0
longitude_imu = 0
latitude_imu = 0
version = 0
readreg = 0
key = 0
flag = 0
iapflag = 0
global recordflag
buff = {}
calibuff = list()
global recordbuff
angularVelocity = [0, 0, 0]
acceleration = [0, 0, 0]
magnetometer = [0, 0, 0]
angle_degree = [0, 0, 0]
mag_offset = [0, 0, 0]
mag_range = [0, 0, 0]
offset_mag = 0
offset_mag2 = 0
global wt_imu
baudlist = [4800, 9600, 19200, 38400, 57600, 115200, 230400, 460800]
motor = Driver_Sensors()
time_ant = rospy.Time.now()
theta = 0
angular_vel_ant = 0

def recordThread():
    global recordflag, recordbuff
    recordflag = 1
    recordbuff = ''
    recordname = time.strftime("%Y%m%d%H%M%S", time.localtime()) + '.txt'
    fd = open(recordname, 'w+')
    print('begin recording file name is {}'.format(recordname))
    while recordflag:
        if len(recordbuff):
            fd.write(recordbuff)
            recordbuff = ''
        else:
            time.sleep(1)

    fd.close()
    print("stop recording")

def init_imu():
    global readreg, flag, calibuff, wt_imu, iapflag, mag_offset, mag_range, version, recordflag, baudlist, offset_mag, offset_mag2

    print('----------------INICIALIZANDO IMU--------------------')
    angle_radian = [angle_degree[i] * math.pi / 180 for i in range(3)]
    offset_mag = angle_radian[2]

    (r,p,y) = (angle_radian[0], angle_radian[1], angle_radian[2])

    Mx = magnetometer[0]*math.cos(p) + magnetometer[2]*math.sin(p)
    My = magnetometer[0]*math.sin(r)*math.sin(p) + magnetometer[1]*math.cos(r) - magnetometer[2]*math.sin(r)*math.cos(p) 
    
    offset_mag2 = -math.atan2(My,Mx)

def callback(data):
    global readreg, flag, calibuff, wt_imu, iapflag, mag_offset, mag_range, version, recordflag, baudlist, offset_mag, offset_mag2
    unlock_imu_cmd = b'\xff\xaa\x69\x88\xb5'
    reset_magx_offset_cmd = b'\xff\xaa\x0b\x00\x00'
    reset_magy_offset_cmd = b'\xff\xaa\x0c\x00\x00'
    reset_magz_offset_cmd = b'\xff\xaa\x0d\x00\x00'
    enter_mag_cali_cmd = b'\xff\xaa\x01\x09\x00'
    exti_cali_cmd = b'\xff\xaa\x01\x00\x00'
    save_param_cmd = b'\xff\xaa\x00\x00\x00'
    read_mag_offset_cmd = b'\xff\xaa\x27\x0b\x00'
    read_mag_range_cmd = b'\xff\xaa\x27\x1c\x00'
    reboot_cmd = b'\xff\xaa\x00\xff\x00'
    reset_mag_param_cmd = b'\xff\xaa\x01\x07\x00'
    set_rsw_demo_cmd = b'\xff\xaa\x02\x1f\x00'  #output time acc gyro angle mag

    print('callback')
    print(data)
    if "calibrate" in data.data:
        calib_aux = 1
        while(calib_aux and not rospy.is_shutdown()):
            print("----------------ENTRA A CALIBRAR---------------------------")
            wt_imu.write(unlock_imu_cmd)
            time.sleep(0.1)
            wt_imu.write(reset_magx_offset_cmd)
            time.sleep(0.1)
            wt_imu.write(reset_magy_offset_cmd)
            time.sleep(0.1)
            wt_imu.write(reset_magz_offset_cmd)
            time.sleep(0.1)
            wt_imu.write(reset_mag_param_cmd)
            time.sleep(0.1)
            wt_imu.write(enter_mag_cali_cmd)
            time.sleep(0.1)
            flag = 1
            calibuff = []
            mag_offset = [0, 0, 0]
            mag_range = [500, 500, 500]
            control_publisher.publish(Float32MultiArray(data=[-4.15, -4.15, 4.15, 4.15]))
            rospy.sleep(7)
            control_publisher.publish(Float32MultiArray(data=[-2.15, -2.15, 2.15, 2.15]))
            rospy.sleep(7)
            control_publisher.publish(Float32MultiArray(data=[0, 0, 0, 0]))
            rospy.sleep(1)
            flag = 0
            wt_imu.write(unlock_imu_cmd)
            time.sleep(0.1)
            wt_imu.write(exti_cali_cmd)
            time.sleep(0.1)
            wt_imu.write(save_param_cmd)
            time.sleep(1)
            readreg = 0x0b
            wt_imu.write(read_mag_offset_cmd)
            time.sleep(1)
            readreg = 0x1c
            wt_imu.write(read_mag_range_cmd)
            time.sleep(1)
            datalen = len(calibuff)
            print('cali data {}'.format(datalen))
            r = list()
            if datalen > 0:
                for i in range(datalen):
                    tempx = ((calibuff[i][0] - mag_offset[0])*2/float(mag_range[0]))
                    tempy = ((calibuff[i][1] - mag_offset[1])*2/float(mag_range[1]))
                    temp = tempx*tempx+tempy*tempy-1
                    r.append(abs(temp))
                sumval = sum(r)
                r_n = float(sumval)/datalen
                if r_n < 0.05:
                    print('magnetic field calibration results are very good')
                    angle_radian = [angle_degree[i] * math.pi / 180 for i in range(3)]
                    qua = quaternion_from_euler(angle_radian[0], angle_radian[1], angle_radian[2])
                    imu_msg.orientation.x = qua[0]
                    imu_msg.orientation.y = qua[1]
                    imu_msg.orientation.z = qua[2]
                    imu_msg.orientation.w = qua[3]
                    #angulo_r.publish(imu_msg)
                    calib_aux = 0
                elif r_n < 0.1:
                    print('magnetic field calibration results are good')
                else :
                    print('magnetic field calibration results is bad, please try again')
        rospy.sleep(1)
        angle_radian = [angle_degree[i] * math.pi / 180 for i in range(3)]
        offset_mag = angle_radian[2]

        (r,p,y) = (angle_radian[0], angle_radian[1], angle_radian[2])

        Mx = magnetometer[0]*math.cos(p) + magnetometer[2]*math.sin(p)
        My = magnetometer[0]*math.sin(r)*math.sin(p) + magnetometer[1]*math.cos(r) - magnetometer[2]*math.sin(r)*math.cos(p) 
        
        offset_mag2 = -math.atan2(My,Mx)

    elif "reset" in data.data:
        angle_radian = [angle_degree[i] * math.pi / 180 for i in range(3)]
        offset_mag = angle_radian[2]
        
        (r,p,y) = (angle_radian[0], angle_radian[1], angle_radian[2])

        Mx = magnetometer[0]*math.cos(p) + magnetometer[2]*math.sin(p)
        My = magnetometer[0]*math.sin(r)*math.sin(p) + magnetometer[1]*math.cos(r) - magnetometer[2]*math.sin(r)*math.cos(p) 
        
        offset_mag2 = -math.atan2(My,Mx)

    elif "version" in data.data:
        print('sensor version is {}'.format(version))
    elif "begin" in data.data:
        record_thread = threading.Thread(target = recordThread)
        record_thread.start()
    elif "stop" in data.data:
        recordflag = 0
    elif "rate" in data.data:
        ratelist = [0.2, 0.5, 1,2,5,10,20,50,100,125,200]
        try:
            val = data.data[4:]
            rate = float(val)
            for i in range(len(ratelist)):
                if rate == ratelist[i]:
                    print('chage {} rate'.format(rate))
                    val = i + 1
                    cmd = bytearray(5)
                    cmd[0] = 0xff
                    cmd[1] = 0xaa
                    cmd[2] = 0x03
                    cmd[3] = val
                    cmd[4] = 0x00
                    wt_imu.write(unlock_imu_cmd)
                    time.sleep(0.1)
                    wt_imu.write(cmd)
        except Exception as e:
            print(e)
    elif "baud" in data.data:
        try:
            val = data.data[4:]
            baud = float(val)
            #baud = 115200.0
            for i in range(len(baudlist)):
                if baud == baudlist[i]:
                    val = i + 1
                    cmd = bytearray(5)
                    cmd[0] = 0xff
                    cmd[1] = 0xaa
                    cmd[2] = 0x04
                    cmd[3] = val
                    cmd[4] = 0x00
                    wt_imu.write(unlock_imu_cmd)
                    time.sleep(0.1)
                    wt_imu.write(cmd)
                    time.sleep(0.1)
                    wt_imu.baudrate = baud
        except Exception as e:
            print(e)
    elif "rsw" in data.data:
        wt_imu.write(unlock_imu_cmd)
        time.sleep(0.1)
        wt_imu.write(set_rsw_demo_cmd)
        time.sleep(0.1)


def thread_job():
    print("thread run")
    rospy.spin()

def AutoScanSensor():
    global wt_imu, baudlist
    try:
        for baud in baudlist:
            read_cmd = '\xff\xaa\x27\x00\x00'.encode("utf-8")
            wt_imu.baudrate = baud
            wt_imu.flushInput()
            wt_imu.write(read_cmd)
            time.sleep(0.2)
            buff_count = wt_imu.inWaiting()
            if buff_count >= 11:
                buff_data = wt_imu.read(buff_count)
                val = bytearray(buff_data)
                for i in range(len(val)):
                    if val[i] == 0x55:
                        sumval = sum(val[i:i+10])
                        if sumval == val[i+10]:
                            print('{} baud find sensor'.format(baud))
                            return

    except Exception as e:
        print("exception:" + str(e))
        print("imu loss of connection, poor contact, or broken wire")
        #exit(0)
  
#global tprev
#tprev = time.time()
if __name__== '__main__':
    #while True:
    vc = Driver_Sensors() 
    freq = rospy.get_param("frecuency", 100)
    period = 1.00/freq
    print("Periodo: " + str(period))
    global recordflag, recordbuff, wt_imu
    recordflag = 0
    recordbuff = list()
    wt_imu = serial.Serial()
    python_version = platform.python_version()[0]

    find_ttyUSB()
    #rospy.init_node("imu")
    port = rospy.get_param("port2", "/dev/ttyUSB2")   #THS1
    baudrate = rospy.get_param("baud2", 115200)
    # baudrate = 115200
    print("IMU Type: Normal Port:%s baud:%d" %(port,baudrate))
    imu_msg = Imu()
    angle_msg = Float64()
    mag_msg = MagneticField()
    location_msg = NavSatFix()
    rospy.Subscriber("/calibrate_imu", String, callback) #接受topic名称
    add_thread = threading.Thread(target = thread_job)
    add_thread.start()
    aux = False
    try:      
        imu_encoder_pub = rospy.Publisher("/imu_encoder", imu_encoder, queue_size=2)
        mag_pub = rospy.Publisher("wit/mag", MagneticField, queue_size=10)
        control_publisher = rospy.Publisher("wheel_setpoint", Float32MultiArray, queue_size=10)
        tprev = time.time()
        t2 = time.time()
        
        wt_imu = serial.Serial(port=port, baudrate=baudrate, timeout=20)
        if wt_imu.isOpen():
            rospy.loginfo("\033[32mSerial port enabled successfully...\033[0m")
        else:
            wt_imu.open()
            rospy.loginfo("\033[32mSerial port enabled successfully...\033[0m")
        #
        while not rospy.is_shutdown():
            ti = time.time()
            
            while (ti-t2)<period:
                ti = time.time()
            #print(ti-t2)
            t2 = time.time()
            
            try:
                buff_count = wt_imu.inWaiting()
                if buff_count > 0 and iapflag == 0:
                    buff_data = wt_imu.read(buff_count)
                    if recordflag:
                        recordbuff = recordbuff + buff_data
                    #print("entra")
                    for i in range(0, buff_count):
                        handleSerialData(buff_data[i])
                        
                    if aux == False:
                        init_imu()
                        aux = True
            except Exception as e:
                print("exception:" + str(e))
                print("imu loss of connection, poor contact, or broken wire")
                #exit(0)
            vc.publish_speed()
            
        rospy.spin()	
    except Exception as e:
        rospy.logerr(str(e))
        #
        print(e)
        rospy.loginfo("\033[31mFailed to open the serial port\033[0m")
        exit(0)
        #
    finally:
        sys.exit()
    
            
