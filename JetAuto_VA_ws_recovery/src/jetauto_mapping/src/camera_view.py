#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import cv_bridge
import numpy as np
from sensor_msgs.msg import Image
from apriltag_ros.msg import AprilTagDetectionArray

class TagAnnotator:
    def __init__(self):
        rospy.init_node('tag_annotator')

        self.bridge = cv_bridge.CvBridge()
        self.latest_detections = []  # Guardar las detecciones recientes

        # Parámetros de la cámara 
        self.fx = 1103.02
        self.fy = 1085.81
        self.cx = 310.221
        self.cy = 189.907

        self.image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.image_callback)
        self.tag_sub = rospy.Subscriber('/tag_detections', AprilTagDetectionArray, self.tag_callback)

    def tag_callback(self, msg):
        self.latest_detections = msg.detections

    def image_callback(self, msg):
        # Convertir imagen ROS -> OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Dibujar detecciones en la imagen
        for detection in self.latest_detections:
            tag_id = detection.id[0]
            position = detection.pose.pose.pose.position
            x, y, z = position.x, position.y, position.z

            if z <= 0:  # Seguridad para evitar divisiones por cero
                continue

            # Proyección de la posición 3D a 2D imagen
            u = int((self.fx * x) / z + self.cx)
            v = int((self.fy * y) / z + self.cy)

            distance = np.sqrt(x**2 + y**2 + z**2)  # Distancia real 3D

            text = "ID:{0} Dist:{1:.2f}m".format(tag_id, distance)

            # Dibujar en la imagen
            if 0 <= u < cv_image.shape[1] and 0 <= v < cv_image.shape[0]:  # Solo si está dentro de la imagen
                cv2.putText(cv_image, text, (u-50, v-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                cv2.circle(cv_image, (u, v), 5, (0,0,255), -1)

        # Mostrar imagen
        cv2.imshow("Camera View with AprilTags", cv_image)
        cv2.waitKey(1)

if __name__ == '__main__':
    annotator = TagAnnotator()
    rospy.spin()



