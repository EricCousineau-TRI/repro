#!/usr/bin/env python2

from __future__ import division
import time

import numpy as np

import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image, CameraInfo
from tf2_ros import StaticTransformBroadcaster


class SimpleCamera(object):
    def __init__(self, frame_id, info_topic):
        self.frame_id = frame_id
        self.img_pub = rospy.Publisher(frame_id, Image, queue_size=10)
        self.info_pub = rospy.Publisher(info_topic, CameraInfo, queue_size=10)

    def publish(self, image):
        h, w = image.shape[:2]
        # Make up intriniscs.
        fov_y = np.pi / 4
        fy = h / (2 * np.tan(fov_y / 2))
        fx = fy  # Assume equal
        K = np.array([
            [fx, 0, w / 2],
            [0, fy, h / 2],
            [0, 0, 1]])
        P = np.hstack((K, np.array([[0, 0, 0]]).T))
        info_msg = CameraInfo()
        info_msg.header.frame_id = self.frame_id
        info_msg.width = w
        info_msg.height = h
        info_msg.K[:] = K.flat  # row-major
        info_msg.R[:] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info_msg.D = [0., 0., 0., 0., 0.]
        info_msg.P[:] = P.flat  # row-major
        info_msg.distortion_model = "plumb_bob"
        img_msg = CvBridge().cv2_to_imgmsg(image, "rgb8")
        img_msg.header.frame_id = self.frame_id
        self.img_pub.publish(img_msg)
        self.info_pub.publish(info_msg)


def main():
    rospy.init_node("repro")
    tf_broadcast = StaticTransformBroadcaster()
    cameras = []
    for i in range(6):
        camera = SimpleCamera(
            "/camera/{}/image_rect_color".format(i),
            "/camera/{}/camera_info".format(i))
        cameras.append(camera)
    time.sleep(0.2)  # Let publishers boot up

    def img_update(_):
        for i, camera in enumerate(cameras):
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            image[:] = (i + 1) / len(cameras) * 255
            camera.publish(image)

    def tf_update(_):
        for i, camera in enumerate(cameras):
            msg = TransformStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "world"
            msg.child_frame_id = camera.frame_id
            p = msg.transform.translation
            p.x = i * 0.5
            q = msg.transform.rotation
            q.w = 1
            tf_broadcast.sendTransform(msg)

    pub_timer = rospy.Timer(rospy.Duration(1), img_update)
    tf_timer = rospy.Timer(rospy.Duration(0.1), tf_update)
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[ Done ]")
