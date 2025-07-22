#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import rospy

from jetauto_interfaces.msg import motors_registers   #ros messages
#from driverMotors.motors import EncoderMotorController 
from driverMotors import EncoderMotorController

ENCODER_MOTOR_MODULE_ADDRESS = 0x34
LED_PIN = 18

class JetAutoMotorTestWC:
    def __init__(self):
        rospy.init_node('CMD_Motors', anonymous=False)   

        rospy.Subscriber('/jetauto_wheels_cmd', motors_registers, self.test_callback)  

        self.motor_controller = EncoderMotorController(1, pulse_per_cycle = 3960)
        self.vx = 0
        self.vy = 0
        self.va = 0 			       
        self.w1 = 0
        self.w2 = 0
        self.w3 = 0
        self.w4 = 0
       
        self.machine_type = os.environ.get('MACHINE_TYPE')

    def enc_callback(self, msg):
        self.w1 = msg.w1
        self.w2 = msg.w2
        self.w3 = msg.w3
        self.w4 = msg.w4

    def test_callback(self, msg):
        m1 = msg.motor1
        m2 = msg.motor2
        m3 = msg.motor3
        m4 = msg.motor4
        self.motor_controller.set_speed((-m1, m2, -m3, m4))
                     
if __name__ == "__main__":
    try:
        node = JetAutoMotorTestWC()
        rospy.spin()
    except Exception as e:
        rospy.logerr(str(e))
    finally:
        node.motor_controller.set_speed((0, 0, 0, 0))
        sys.exit()
