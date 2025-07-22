#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist

# Variables globales
MAX_LIN_VEL = 0.5  # Velocidad lineal máxima
MAX_ANG_VEL = 0.5  # Velocidad angular máxima
linear_axis = 2    # Eje del joystick para avanzar/retroceder (normalmente eje Y del stick izquierdo)
angular_axis = 1   # Eje del joystick para girar (normalmente eje X del stick izquierdo)

# Publicador de velocidad
pub = rospy.Publisher('/jetauto_wheels_cmd', Twist, queue_size=10)

def joy_callback(msg):
    """
    Callback que recibe los datos del joystick y los convierte en velocidades
    """
    twist = Twist()

    # Leer ejes del joystick
    if msg.buttons[0] == 1 :
        twist.linear.x = 1 * MAX_LIN_VEL
    else: 
        twist.linear.x = msg.axes[linear_axis] * MAX_LIN_VEL
    
    twist.angular.z = msg.axes[angular_axis] * MAX_ANG_VEL

    # Publicar velocidad en /jetauto_cmd_vel
    pub.publish(twist)

def main():
    rospy.init_node("jetauto_joy_control", anonymous=True)

    # Suscribirse al tópico del joystick
    rospy.Subscriber("/joy", Joy, joy_callback)

    rospy.loginfo("Control de JetAuto con joystick iniciado.")
    rospy.spin()  # Mantener el programa ejecutándose

if __name__ == "__main__":
    main()
