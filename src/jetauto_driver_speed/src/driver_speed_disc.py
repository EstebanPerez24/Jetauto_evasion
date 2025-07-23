#!/usr/bin/env python

import os
import sys
import rospy
#import time
from std_msgs.msg import Float32MultiArray, Header
from jetauto_interfaces.msg import imu_encoder
from jetauto_interfaces.msg import motor_testWC   #ros messages

class WheelSpeedController:
    def __init__(self):
        rospy.init_node("wheel_speed_controller", anonymous=False)
        
        # Parameters for the PID controller
        self.Kp = rospy.get_param("~Kp", 6.1)
        self.Ki = rospy.get_param("~Ki", 5.75)
        self.Kd = rospy.get_param("~Kd", 0.00)
        
        self.e_ant = [0,0,0,0]
        self.control_outputs_ant = [0, 0, 0, 0]
        
        # Setpoint for the wheel speed
        self.setpoint2 = Float32MultiArray(data = [0.0, 0.0, 0.0, 0.0])
        #Calculo constantes para el controlador
        self.Ts = 1.0/90.909       
        #if not self.setpoint.data:
        #    rospy.logwarn("Setpoint no recibido aun")
        #    return
        
        self.A0 = self.Kp + self.Ki * self.Ts / 2 + 2 * self.Kd / self.Ts
        self.A1 = -self.Kp + self.Ki * self.Ts / 2 - 2 * self.Kd / self.Ts
        self.A2 = self.Kd / self.Ts  # Additional term needed for proper derivative calculation
        
        
        
        # Subscriber for the wheel speed
        
        rospy.Subscriber("/imu_encoder", imu_encoder, self.control_callback)
        # Publisher for the control output
        self.control_publisher = rospy.Publisher("jetauto_wheels_cmd", motor_testWC, queue_size=10)
        rospy.Subscriber("wheel_setpoint", Float32MultiArray, self.setpoint_callback)
        
        

        
    def control_callback(self, msg):
        #print("CONTROL")
        # Read current wheel speeds
        current_speeds = [msg.w1, msg.w2, msg.w3, msg.w4]

        # Ensure we have the correct number of wheel speeds
        if len(current_speeds) != 4:
            rospy.logwarn("Received wheel speed data does not match expected number of wheels (4).")
            return
        
        control_outputs = [0,0,0,0]
        
        # Compute control outputs for each wheel
        for i in range(4):
            error = self.setpoint2.data[i] - current_speeds[i]
            # Compute control output with the Tustin method
            control_outputs[i] = self.control_outputs_ant[i] + self.A0 * error + self.A1 * self.e_ant[i] + self.A2 * (error - self.e_ant[i])
            if (control_outputs[i]>=80):
                control_outputs[i] = 80
            if (control_outputs[i]<=-80):
                control_outputs[i] = -80
            if self.setpoint2.data[i] == 0:
                control_outputs[i] = 0
        
            # Update previous values
            self.control_outputs_ant[i] = control_outputs[i]
            self.e_ant[i] = error
            
        # Create a message for the control outputs
        #print(self.e_ant)
        self.control_publisher.publish(int(control_outputs[0]), int(control_outputs[1]), int(control_outputs[2]), int(control_outputs[3]))

        
    def setpoint_callback(self, msg):
        #print("entra")

        self.setpoint2 = msg
        #print("sale")
        
    def stop(self):
        self.control_publisher.publish(int(0), int(0), int(0), int(0))

if __name__ == "__main__":
    try:
        node = WheelSpeedController()
        rospy.spin()
        if rospy.is_shutdown():
            node.stop()
    except Exception as e:
        rospy.logerr(str(e))
    finally:
        node.stop()
        sys.exit()
