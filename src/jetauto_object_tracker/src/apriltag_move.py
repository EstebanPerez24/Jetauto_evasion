#!/usr/bin/env python3
import rospy
import math
import tf.transformations as tf
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState

class AprilTagMoverL:
    def __init__(self):
        rospy.init_node("apriltag_L_mover")

        rospy.wait_for_service('/gazebo/set_model_state')
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        self.tag_name = "Apriltag36_11_00000"
        self.z = 0.1

        self.x = 1.0
        self.y = 0.0
        self.yaw = 0.0

        self.linear_speed = 0.5 # m/s
        self.angular_speed = 0.5  # rad/s

        self.rate = rospy.Rate(10)  # 10 Hz

        self.state = "forward_x"

        self.set_pose(self.x, self.y, self.yaw)

    def set_pose(self, x, y, yaw):
        quat = tf.quaternion_from_euler(0, 1.57, yaw)

        state = ModelState()
        state.model_name = self.tag_name
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = self.z
        state.pose.orientation.x = quat[0]
        state.pose.orientation.y = quat[1]
        state.pose.orientation.z = quat[2]
        state.pose.orientation.w = quat[3]

        try:
            self.set_state(state)
        except rospy.ServiceException as e:
            rospy.logerr("Error al mover el tag: %s", e)

    def run(self):
        while not rospy.is_shutdown():
            if self.state == "forward_x":
                self.x += self.linear_speed * 0.1
                if self.x >= 2.0:
                    self.x = 2.0
                    self.state = "rotate"
            elif self.state == "rotate":
                self.yaw += self.angular_speed * 0.1
                if self.yaw >= 1.57:
                    self.yaw = 1.57
                    self.state = "forward_y"
            elif self.state == "forward_y":
                self.y += self.linear_speed * 0.1
                if self.y >= 2.0:
                    self.y = 2.0
                    self.state = "done"
            elif self.state == "done":
                rospy.loginfo_once("AprilTag llegó a (2, 2)")
                pass

            self.set_pose(self.x, self.y, self.yaw)
            self.rate.sleep()

if __name__ == "__main__":
    try:
        mover = AprilTagMoverL()
        mover.run()
    except rospy.ROSInterruptException:
        pass
