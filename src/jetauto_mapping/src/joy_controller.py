#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import math
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32MultiArray

class ManualJoystickControl:
    def __init__(self):
        rospy.init_node("manual_joystick_control", anonymous=False)

        # Parámetros físicos del JetAuto
        self.r = 0.0485
        self.lx = 0.0975
        self.ly = 0.103
        self.max_w = 9.0  # máximo absoluto en rad/s

        # Escalas de velocidad (ajustables)
        self.v_scale = 0.5
        self.w_scale = 0.5

        # Últimos valores del joystick
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0

        # Publicador y suscriptor
        self.pub = rospy.Publisher("/wheel_setpoint", Float32MultiArray, queue_size=10)
        rospy.Subscriber("/joy", Joy, self.joy_callback)

        # Tasa de publicación
        self.rate = rospy.Rate(10)  # 10 Hz
        self.run_loop()

    def joy_callback(self, msg):
        # Leer ejes del joystick
        self.vx = msg.axes[1] * self.v_scale    # Eje X (avance)
        self.vy = msg.axes[0] * self.v_scale    # Eje Y (lateral)
        self.wz = msg.axes[2] * self.w_scale  # Rotación combinada

    def run_loop(self):
        while not rospy.is_shutdown():
            # Jacobiano inverso (idéntico al de PoseControl)
            th1 = math.pi / 4
            r2 = math.sqrt(2)
            a = self.lx + self.ly
            J_inv = np.array([
                [r2 * math.cos(th1),  r2 * math.sin(th1), -a],
                [r2 * math.sin(th1), -r2 * math.cos(th1), -a],
                [r2 * math.cos(th1),  r2 * math.sin(th1),  a],
                [r2 * math.sin(th1), -r2 * math.cos(th1),  a]
            ])

            # Vector de velocidades
            u = np.array([[self.vx], [self.vy], [self.wz]])
            w = (1 / self.r) * np.dot(J_inv, u)
            w_clipped = np.clip(w.flatten(), -self.max_w, self.max_w)

            # Publicar mensaje como Float32MultiArray
            msg_out = Float32MultiArray()
            msg_out.data = w_clipped.tolist()
            self.pub.publish(msg_out)

            self.rate.sleep()

if __name__ == "__main__":
    try:
        ManualJoystickControl()
    except rospy.ROSInterruptException:
        pass

