#!/usr/bin/env python3
import os
import smbus2
#import threading
#import struct
import rospy
import math
#from locale import currency

ENCODER_MOTOR_MODULE_ADDRESS = 0x34

class EncoderMotorController:
    def __init__(self, i2c_port, pulse_per_cycle = 3960, motor_type=3):
        self.i2c_port = i2c_port
        self.machine_type = os.environ.get('MACHINE_TYPE')
        self.pulse_per_cycle = pulse_per_cycle										# pulsos por vuelta del encoder 
        with smbus2.SMBus(self.i2c_port) as bus:
            bus.write_i2c_block_data(ENCODER_MOTOR_MODULE_ADDRESS, 20, [motor_type, ])					# configuracion de tipo de motor, tipo 3 para 	motor con encoder

    def set_speed(self, speed, motor_id=None, offset=0):				# configura la velocidad de las ruedas
        w_write = [0,0,0,0]
        print("ADIOS")
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
                        bus.write_i2c_block_data(ENCODER_MOTOR_MODULE_ADDRESS, 50 + new_id, [sp, ])		# escribir a partir del registro 51 la velocidad de las ruedas
                        #w_write[i] = sp*math.pi*2/(self.pulse_per_cycle*0.01)      				# calcula la velocidad que se escribe
                    except BaseException as e:
                        print(e)
                        bus.write_i2c_block_data(ENCODER_MOTOR_MODULE_ADDRESS, 50 + new_id, [sp, ])
                        #w_write[i] = sp*math.pi*2/(self.pulse_per_cycle*0.01)
                #print("WRITE:    w1: "+str(w_write[0])+" w2: "+str(w_write[1])+" w3: "+str(w_write[2])+" w4: "+str(w_write[3]))
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
                        #i = motor_id-1
                        #w_write[i] = sp*math.pi*2/(self.pulse_per_cycle*0.01)
                        #print('WRITE:    w'+str(i+1)+': '+str(w_write[i]))
                    except BaseException as e:
                        print(e)
                        bus.write_i2c_block_data(ENCODER_MOTOR_MODULE_ADDRESS, 50 + motor_id, [speed, ])
                        #i = motor_id-1
                        #w_write[i] = sp*math.pi*2/(self.pulse_per_cycle*0.01)
                        #print('WRITE:    w'+str(i+1)+': '+str(w_write[i]))
                else:
                    raise ValueError("Invalid motor id")				# en caso de ingresar un id de motor fuera de rango

