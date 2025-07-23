#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float32MultiArray
from simple_pid import PID
from jetauto_interfaces.msg import motors_registers   #ros messages
from jetauto_interfaces.msg import imu_encoder
from std_msgs.msg import Float32

class WheelSpeedController:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('Driver_Speed', anonymous=False)

        # Parameters for the PID controller
        self.Kp = rospy.get_param("~Kp", 5.1)
        self.Ki = rospy.get_param("~Ki", 4.75)
        self.Kd = rospy.get_param("~Kd", 0.0)

        # Setpoint for the wheel speed
        self.setpoint = rospy.get_param("~setpoint", [0.0, 0.0, 0.0, 0.0])  # Example setpoints for four wheels

        # PID controllers for each wheel
        self.pid_controllers = [PID(self.Kp, self.Ki, self.Kd, setpoint=sp) for sp in self.setpoint]

        # Subscriber for the wheel speed
        rospy.Subscriber("/imu_encoder", imu_encoder, self.control_callback)
        rospy.Subscriber("/wheel_setpoint", Float32MultiArray, self.setpoint_callback)

        # Publisher for the control output
        self.control_publisher = rospy.Publisher("/jetauto_wheels_cmd", motors_registers, queue_size=1)
        self.new_setpoints = Float32MultiArray(data = [0, 0, 0, 0])
        #self.file = open("/home/jetauto/JetAuto_VA_ws/src/Datos_Graficas/PIDWheels2.txt","w")
        #self.file.write("t\tsp\tw1\tw2\tw3\tw4\n")
        #self.file.close()
        self.t1 = rospy.get_rostime()

    def control_callback(self, msg):
        # Read current wheel speeds
        current_speeds = [msg.w1, msg.w2, msg.w3, msg.w4]
        time_now = msg.imu.header.stamp
        #hola = open("/home/jetauto/JetAuto_VA_ws/src/Datos_Graficas/PIDWheels2.txt","a")
        #self.file.write(w1+"\t")
        #hola.write(str((time_now-self.t1).to_sec())+"\t")
        #hola.write(str(self.new_setpoints.data[0])+"\t"+str(current_speeds[0])+"\t"+str(current_speeds[1])+"\t"+str(current_speeds[2])+"\t"+str(current_speeds[3])+"\n")
        #hola.close()

        # Ensure we have the correct number of wheel speeds
        if len(current_speeds) != 4:
            rospy.logwarn("Received wheel speed data does not match expected number of wheels (4).")
            return

        # Compute control outputs for each wheel
        control_outputs = [pid(current_speed) for pid, current_speed in zip(self.pid_controllers, current_speeds)]
        for i in range(4):
            if (control_outputs[i]>=80):
                control_outputs[i] = 80
            if (control_outputs[i]<=-80):
                control_outputs[i] = -80
            if self.new_setpoints.data[i] == 0:
                control_outputs[i] = 0

        # Create a message for the control outputs
        #control_msg = motor_testWC()
        # Publish the control outputs
        self.control_publisher.publish(int(control_outputs[0]), int(control_outputs[1]), int(control_outputs[2]), int(control_outputs[3]))

    def setpoint_callback(self, msg):
        # Update the setpoints for each PID controller
        self.new_setpoints = msg

        # Ensure we have the correct number of setpoints
        if len(self.new_setpoints.data) != 4:
            #rospy.logwarn("Received setpoint data does not match expected number of wheels (4).")
            return

        for pid, sp in zip(self.pid_controllers, self.new_setpoints.data):
            pid.setpoint = sp

        #rospy.loginfo(f"Setpoints updated to: {self.new_setpoints}")
    def run(self):
        rospy.spin()
    def stop(self):
        self.control_publisher.publish(int(0), int(0), int(0), int(0))

if __name__ == "__main__":
    try:
        controller = WheelSpeedController()
        controller.run()
        if rospy.is_shutdown():
              controller.stop()
    except rospy.ROSInterruptException:
        pass

