#! /usr/bin/env python
# -*- coding: utf-8-*-
# Software License Agreement (BSD License)
#
# Copyright (c) 2019, DEEPCALCULATION, Inc.
# All rights reserved.
#
# Author: mgl  <hitmgl@163.com>
# 代码用途：使用乐视双目摄像头通过目标识别和pid控制来巡线

import cv2
import numpy as np
import csv
import datetime
import os
import uuid
import sys
import time


import serial
import threading
import time
import math
import signal
from keras.models import load_model
from keras import backend as K
from box import to_minmax
from box import BoundBox, nms_boxes, boxes_to_array, draw_scaled_boxes
from infer_detector_pc import prepare_image, predict, labels, get_lable_score
from threading import Thread

import rospy
import tf
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError

class image_converter:

    def __init__(self):

        # 找到左边线和右边线标志位
        self.findLeftLine = False
        self.findRightLine = False
        # pid控制的参数
        self.pid_p = 0.006
        self.pid_d = 0.0007  # mini is 0.05
        # 偏差和上一次的偏差
        self.lastErr = 0
        self.lineErr = 0


        # 道路的像素宽度初始值，小车检测到两根边线后会更新此值
        self.width = 530
        self.lastWidth = 0
        self.findNoLine = False

        # 取图像中的哪一行做为边线检测
        self.detectLine = 320
        self.emptydetectLine = 360
        self.emptyLines = 20

        self.rotationSpeed = 0.0
        self.linearSpeed = 0.0
        self.forwardSpeed = 0.2

        # canny边缘检测的高低阈值
        self.thresholdCannyL = 120
        self.thresholdCannyH = 240

        self.debug = True
        self.debugType = 1

        self.enable_turn = False
        self.distanceThr = 0.8
        self.distanceRatio = 1.0
        self.distanceSign = 0.35
        self.distanceTraffic = 1.0
        self.enable_distance_detect = False
        self.enable_laser_detect = False
        self.min_laser_distance = 0.3
        self.rotationAngle = 90.0

        self.kernel_size = 3
        self.stop = False
        self.laser_stop_robot = False
        self.findCorner = False
        self.reachDelayTime = False
        self.close_pid = False
        self.recordEncoder = False
        self.yawCurrent = 0
        self.yawStart = 0
        self.yawTarget = 0
        self.yawError = 0
        self.yawLastError = 0

        self.pid_p_yaw = 0.85
        self.pid_d_yaw = 0.04
        self.cornerDir = 0

        self.msg = Twist()
        self.pub = rospy.Publisher('/cmd_vel', Twist)


        self.model_path = 'C:/Users/admin/Desktop/src/YOLO_best_mAP.h5'
        #/home/deepcar/catkin_ws/src/object_detection_mobilenet/src/traffic_signs_YOLO_best_mAP.h5
        self.modelInit = False
        self.result = []
        self.depthValid = False
        self.distance = 0
        self.currentLabel = ''
        self.lableScore = 0
        self.no_object = True
        self.result_image = None
        self.gray = None

        self.initParam()




        # 使用CvBridge将ros格式的图像转为opecv格式的图像
        self.bridge = CvBridge()
        # 双目相机的rbg图主题，此处不需要用到深度图
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.Imagecallback, queue_size=1,
                                          buff_size=2 ** 24)
        self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.Depthcallback, queue_size=1,
                                          buff_size=2 ** 24)
        # 发布压缩图像到手机上
        self.image_pub = rospy.Publisher("/usb_cam/image_raw/compressed", CompressedImage, queue_size=1)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odomCb)
        self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.laserCb)


    def initParam(self):

        self.thresholdCannyL = rospy.get_param("~lowTh", 120)
        self.thresholdCannyH = rospy.get_param("~highTh", 240)
        self.centerLine = rospy.get_param("~centerLine", 350)
        self.detectLine = rospy.get_param("~detectLine", 320)
        self.width = rospy.get_param("~width", 450)
        self.forwardSpeed = rospy.get_param("~forwardSpeed", 0.2)
        self.pid_p = rospy.get_param("~pid_p", 0.004)
        self.pid_d = rospy.get_param("~pid_d", 0.0005)
        self.debugType = rospy.get_param("~debugType", 0)
        self.enable_turn = rospy.get_param("~enable_turn", False)
        self.distanceThr = rospy.get_param("~distanceThr", 0.8)
        self.distanceRatio = rospy.get_param("~distanceRatio", 1.0)
        self.distanceSign = rospy.get_param("~distanceSign", 0.35)
        self.distanceTraffic = rospy.get_param("~distanceTraffic", 1.0)
        self.enable_distance_detect = rospy.get_param("~enable_distance_detect", False)
        self.enable_laser_detect = rospy.get_param("~enable_laser_detect", False)
        self.min_laser_distance = rospy.get_param("~min_laser_distance", 0.3)
        self.rotationAngle = rospy.get_param("~rotationAngle", 90.0)
        print(self.thresholdCannyL,self.thresholdCannyH,self.centerLine,self.detectLine,self.width,self.forwardSpeed,self.pid_p,self.pid_d,self.debugType)

    def setTurnTimer(self):
        self.reachDelayTime = True
        print('Turn Time Out!')

    def setTarget(self, targetYaw):
        self.yawTarget = math.radians(targetYaw)

    def odomCb(self, data):

        quaternion = (
            data.pose.pose.orientation.x,
            data.pose.pose.orientation.y,
            data.pose.pose.orientation.z,
            data.pose.pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        #roll = euler[0]
        #pitch = euler[1]
        self.yawCurrent = euler[2]

    def laserCb(self, data):

        min_distance =  min(min(data.ranges[162:197]), 10)
        if self.enable_laser_detect and min_distance < self.min_laser_distance:
            self.laser_stop_robot = True
            print("obstacle detected! min laser distance is {}".format(self.min_laser_distance) + " ,now stop the robot!")
        else:
            self.laser_stop_robot = False

    def Depthcallback(self, data):

        try:
            # 将订阅的双目相机深度图像转为opencv的图像格式
            depth_image = self.bridge.imgmsg_to_cv2(data, "16UC1")
        except CvBridgeError as e:
            print(e)

        if self.no_object:
            return
        dist_val = [0, 0, 0, 0, 0]
        if len(self.result) != 4:
            return
        x1, y1, x2, y2 = self.result
        if x1 < 0:
            x1 = 0
        if x2 > 639:
            x2 = 639
        if y1 < 0:
            y1 = 0
        if y2 > 479:
            y2 = 479
        width = x2 - x1
        height = y2 - y1

        if width < 20 or height < 20:
            return

        dist_val[0] = float(depth_image[y1 + height / 3, x1 + width / 3]) / 1000
        dist_val[1] = float(depth_image[y1 + height / 3, x1 + 2 * width / 3]) / 1000
        dist_val[2] = float(depth_image[y1 + 2 * height / 3, x1 + width / 3]) / 1000
        dist_val[3] = float(depth_image[y1 + 2 * height / 3, x1 + 2 * width / 3]) / 1000
        dist_val[4] = float(depth_image[y1 + height / 2, x1 + width / 2]) / 1000
        distance_sum = 0.0

        num_depth_points = 5

        for i in range(0, 5):
            if dist_val[i] > 0.4 and dist_val[i] < 10.0:
                distance_sum = dist_val[i] + distance_sum
            else:
                num_depth_points = num_depth_points - 1
        if num_depth_points > 0:
            self.depthValid = True
            self.distance = distance_sum / num_depth_points
            print("distance is {}".format(self.distance))
        else:
            self.depthValid = False
            # print("no valid depth point")

    # 主循环
    def Imagecallback(self, data):
        # 不能在构造函数中初始化，否则会出错
        if self.modelInit == False:
            self.modelInit = True
            self.model = load_model(self.model_path)

        try:
            # 将订阅的双目相机图像转为opencv的图像格式
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        # 将图像转为灰度图

        self.stop = False
        self.currentLabel = ''
        self.lableScore = 0
        img_cp = cv_image.copy()

        # 将图像转为灰度图
        gray_raw = cv2.cvtColor(img_cp, cv2.COLOR_BGR2GRAY)
        # 初始化左右边线都没找到
        self.findLeftLine = False
        self.findRightLine = False
        # 将灰度图高斯模糊
        gaussian_blurred = cv2.GaussianBlur(gray_raw, (self.kernel_size, self.kernel_size), 0)
        # canny边缘检测，只保留图像边缘
        self.gray = cv2.Canny(gaussian_blurred, self.thresholdCannyL, self.thresholdCannyH)

        # ***************************************************************************************************#
        # 目标检测#
        orig_image, input_image = prepare_image(cv_image)
        height, width = orig_image.shape[:2]

        prediction_time, boxes, probs = predict(self.model, input_image, height, width)
        print(prediction_time)
        label_score_list = get_lable_score(probs, labels)
        # print(label_score_list)
        self.result_image = draw_scaled_boxes(orig_image, boxes, probs, labels)

        if self.findCorner:

            for k in range(self.emptydetectLine-self.emptyLines, self.emptydetectLine+self.emptyLines, 1):
                for i in range(320, 640, 1):
                    # 从中间往右边找，如果找到白点，则说明找到了右边线的像素点
                    if self.gray[k][i] == 255:
                        self.findRightLine = True
                        break
                # 从中间往左边找，如果找到白点，则说明找到了左边线的像素点
                for i in range(320, 0, -1):
                    if self.gray[k][i] == 255:
                        self.findLeftLine = True
                        break
            if (not self.findLeftLine) and (not self.findRightLine):
                self.close_pid = True
                self.rotationSpeed = 0
                print("close pid controller!")

            if self.reachDelayTime:
                self.close_pid = True
                if not self.recordEncoder:
                    self.recordEncoder = True
                    self.yawStart = self.yawCurrent
                    if self.cornerDir == 0:
                        self.setTarget(self.rotationAngle);
                    else:
                        self.setTarget(-self.rotationAngle);

                yawDiff = 0
                if (self.yawCurrent - self.yawStart) > math.pi:
                    yawDiff = self.yawCurrent - self.yawStart - 2 * math.pi
                elif (self.yawCurrent - self.yawStart) < -math.pi:
                    yawDiff = self.yawCurrent - self.yawStart + 2 * math.pi
                else:
                    yawDiff = self.yawCurrent - self.yawStart

                self.yawError = self.yawTarget - yawDiff;
                self.linearSpeed = 0;
                self.rotationSpeed = self.pid_p_yaw * self.yawError + self.pid_d_yaw * (self.yawError - self.yawLastError)
                if self.rotationSpeed > 0.3:
                    self.rotationSpeed = 0.3;
                elif self.rotationSpeed < -0.3:
                    self.rotationSpeed = -0.3;

                self.yawLastError = self.yawError;

                # print("yaw cureent is : {}",format(self.yawCurrent))
                # print("yaw start is : {}".format(self.yawStart))
                # print("yaw diff is : {}".format(yawDiff))
                # print("yaw error is : {}".format(self.yawError))

                if abs(self.yawError) < 0.02:

                    print("reach target angle! {}".format(self.yawError))
                    self.distance = 0
                    self.findCorner = False;
                    self.recordEncoder = False;
                    self.reachDelayTime = False;
                    self.close_pid = False;
                    self.rotationSpeed = 0;

                return


        # 将检测的行的下一行置为白色，此处是为了方便在手机app图像上看摄像头前瞻
        for i in range(0, 640):
            self.gray[self.detectLine + 1][i] = 255

        self.findNoLine = False
        for k in range(320, 640, 1):
            # 从中间往右边找，如果找到白点，则说明找到了右边线的像素点
            if self.gray[self.detectLine][k] == 255:
                self.findRightLine = True
                REdge = k
                break
        # 从中间往左边找，如果找到白点，则说明找到了左边线的像素点
        for k in range(320, 0, -1):
            if self.gray[self.detectLine][k] == 255:
                self.findLeftLine = True
                LEdge = k
                break
        # 如果左右边线都找到了，则更新道路的像素宽度
        if self.findRightLine and self.findLeftLine:
            self.width = REdge - LEdge
            if self.width < 400 and self.lastWidth > 400:
                self.width = self.lastWidth
                print("invalid width!")
            #print("find two lines")
        # 如果只找到左边线，则右边线的位置是左边线加道路宽度
        elif self.findLeftLine:
            REdge = LEdge + self.width
            #print("only find the left line")
        # 如果只找到右边线，则左边线的位置是右边线减去道路宽度
        elif self.findRightLine:
            LEdge = REdge - self.width
            #print("only find the right line")
        else:
            self.findNoLine = True
            #print("no line found")
        # 如果两根线都找不到，则小车偏离道路的值使用上一次的左右线的位置来计算
        if not self.findNoLine:
            self.lineErr = 320 - (REdge + LEdge) / 2


        # 找出概率最大的label
        for box, item in zip(boxes, label_score_list):
            if item['score'] > self.lableScore:
                self.lableScore = item['score']
                self.currentLabel = item['label']

        if (self.currentLabel == 'red' or self.currentLabel == 'yellow') and int(self.lableScore * 100) >= 50:
            self.result = box
            if self.enable_distance_detect:
                print(self.distanceTraffic, self.distance)
                if self.distanceTraffic < self.distance:
                    self.stop = False
                else:
                    self.stop = True
            else:
                self.stop = True
            print("find {}".format(self.currentLabel))
        elif self.currentLabel == 'green' and int(self.lableScore * 100) >= 50:
            self.result = box
            self.stop = False
            print("find {}".format(self.currentLabel))
        elif self.currentLabel == 'stop' and int(self.lableScore * 100) >= 50:
            self.result = box
            if self.enable_distance_detect:
                print(self.distanceTraffic, self.distance)
                if self.distanceTraffic < self.distance:
                    self.stop = False
                else:
                    self.stop = True
            else:
                self.stop = True
            print("find {}".format(self.currentLabel))
        elif (self.currentLabel == 'left' and int(self.lableScore * 100) >= 80) or (self.currentLabel == 'right' and int(self.lableScore * 100) >= 80):
            if not self.findCorner:
                self.result = box
                if self.enable_turn:
                    if self.distance > self.distanceThr or self.distance==0:
                        self.findCorner = False
                    else:
                        if self.currentLabel == 'left':
                            self.cornerDir = 0
                        elif self.currentLabel == 'right':
                            self.cornerDir = 1
                        self.findCorner = True
                        delayTime = (self.distance - self.distanceSign) / self.forwardSpeed * self.distanceRatio;
                        timer = threading.Timer(delayTime, self.setTurnTimer)
                        timer.start()
                print("find {}".format(self.currentLabel))

        else:
            self.distance = 0
            self.currentLabel = ''
            self.lableScore = 0

        if self.currentLabel == '':
            self.no_object = True
        else:
            self.no_object = False

        # ***************************************************************************************************

        if not self.close_pid:
            if self.stop:
                self.linearSpeed = 0
                self.rotationSpeed = 0
            else:
                self.linearSpeed = self.forwardSpeed
                # 使用pid控制算法来计算小车的转弯量
                self.rotationSpeed = self.pid_p * self.lineErr + self.pid_d * (self.lineErr - self.lastErr)
        else:
            self.linearSpeed = self.forwardSpeed
            self.rotationSpeed = 0
            print("run winthout pid ")

        if self.laser_stop_robot:
            self.linearSpeed = 0
            self.rotationSpeed = 0

        # 记录上一次小车的偏差，给pid控制用
        self.lastErr = self.lineErr
        self.lastWidth = self.width


# 主循环
def main(args):
    # 初始化ros节点
    rospy.init_node('object_detection_mobilenet', anonymous=True)
    ic = image_converter()
    rate = rospy.Rate(20)
    # 主循环
    while not rospy.is_shutdown():

        ic.msg.linear.x = ic.linearSpeed
        ic.msg.angular.z = ic.rotationSpeed
        ic.pub.publish(ic.msg)
        # 将图像压缩后发送到手机上方便调试

        if ic.debug:
            try:
                if ic.gray is None or ic.result_image is None:
                    continue
                # 将图像压缩
                msg_image = CompressedImage()
                msg_image.header.stamp = rospy.Time.now()
                msg_image.format = "jpg"
                if (ic.debugType == 0):
                    msg_image.data = cv2.imencode(".jpg", ic.gray)[1].tostring()
                else:
                    msg_image.data = cv2.imencode(".jpg", ic.result_image)[1].tostring()
                # 将压缩图像发送到手机app上方便调试
                ic.image_pub.publish(msg_image)
            except CvBridgeError as e:
                print(e)

        rate.sleep()
    rospy.spin()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)



